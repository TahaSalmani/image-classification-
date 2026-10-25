import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cnnclassifier import logger
from cnnclassifier.entity.config_entity import DataIngestionConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.unzip_dir = Path(config.unzip_dir)
        self.local_data_file = Path(config.local_data_file)
        self.manifest_path = self.unzip_dir / "manifest.json"

    def _expected_counts(self):
        return {
            split: {label: int(count) for label, count in labels.items()}
            for split, labels in self.config.expected_counts.items()
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _zip_is_valid(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            with zipfile.ZipFile(path) as archive:
                return archive.testzip() is None
        except (zipfile.BadZipFile, OSError):
            return False

    def _verify_checksum(self, path: Path) -> None:
        expected = (self.config.verify_sha256 or "").strip().lower()
        if not expected:
            return
        actual = self._sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"dataset checksum mismatch: expected {expected}, got {actual}"
            )

    def _manifest(self) -> dict:
        if not self.manifest_path.is_file():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _count_images(self, directory: Path) -> int:
        return sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _actual_counts(self):
        expected = self._expected_counts()
        actual = {}
        for split, labels in expected.items():
            split_dir = self.unzip_dir / split
            if not split_dir.is_dir():
                return None
            present = {entry.name for entry in split_dir.iterdir() if entry.is_dir()}
            if present != set(labels):
                return None
            actual[split] = {
                label: self._count_images(split_dir / label) for label in labels
            }
        return actual

    def _dataset_is_complete(self) -> bool:
        manifest = self._manifest()
        if not manifest:
            return False
        if manifest.get("dataset_slug") != self.config.dataset_slug:
            return False
        return self._actual_counts() == self._expected_counts()

    def _archive_matches_manifest(self) -> bool:
        manifest = self._manifest()
        if not manifest:
            return False
        if not self.local_data_file.is_file():
            return True
        if manifest.get("archive_size") != self.local_data_file.stat().st_size:
            return False
        return manifest.get("archive_sha256") == self._sha256(self.local_data_file)

    def Download_file(self):
        if self._dataset_is_complete():
            logger.info("Prepared dataset already present; skipping download.")
            return

        if self.local_data_file.is_file():
            if not self._zip_is_valid(self.local_data_file):
                logger.warning("Existing archive is corrupt; downloading it again.")
                self.local_data_file.unlink()
            else:
                self._verify_checksum(self.local_data_file)
                logger.info(
                    "Dataset archive already exists at %s", self.local_data_file
                )
                return

        self.unzip_dir.mkdir(parents=True, exist_ok=True)
        staging = self.unzip_dir / ".download"
        kaggle_cli = shutil.which("kaggle")
        if not kaggle_cli:
            raise RuntimeError(
                "kaggle CLI not found; install requirements and add "
                "~/.kaggle/kaggle.json before running data ingestion"
            )

        command = [
            kaggle_cli,
            "datasets",
            "download",
            "-d",
            self.config.dataset_slug,
            "-p",
            str(staging),
        ]
        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            try:
                logger.info(
                    "Downloading %s (attempt %s/%s)",
                    self.config.dataset_slug,
                    attempt,
                    self.config.max_retries,
                )
                subprocess.run(
                    command,
                    check=True,
                    timeout=self.config.download_timeout,
                    stdin=subprocess.DEVNULL,
                )
                last_error = None
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                last_error = error
                if attempt < self.config.max_retries:
                    wait_seconds = 5 * attempt
                    logger.warning(
                        "Download attempt %s failed (%s); retrying in %ss",
                        attempt,
                        error,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)

        if last_error is not None:
            raise RuntimeError(
                f"kaggle download failed after {self.config.max_retries} attempts: "
                f"{last_error}"
            )

        archives = sorted(staging.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(
                f"expected exactly one zip archive in {staging}, found {len(archives)}"
            )
        archive = archives[0]
        if not self._zip_is_valid(archive):
            raise RuntimeError(
                f"downloaded archive failed integrity check: {archive}"
            )
        self._verify_checksum(archive)
        os.replace(archive, self.local_data_file)
        shutil.rmtree(staging, ignore_errors=True)
        logger.info("Dataset downloaded to %s", self.local_data_file)

    @staticmethod
    def _safe_extract(archive_path: Path, destination: Path) -> None:
        destination = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            broken = archive.testzip()
            if broken is not None:
                raise RuntimeError(f"archive member is corrupt: {broken}")
            for member in archive.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RuntimeError(
                        f"archive contains a symlink: {member.filename}"
                    )
                target = (destination / member.filename).resolve()
                if target != destination and destination not in target.parents:
                    raise RuntimeError(
                        f"archive contains an unsafe path: {member.filename}"
                    )
            archive.extractall(destination)

    def _flatten_chest_xray(self, destination: Path) -> None:
        nested = destination / "chest_xray"
        if not nested.is_dir():
            return
        for item in nested.iterdir():
            target = destination / item.name
            if target.exists():
                raise RuntimeError(
                    f"cannot un-nest {item.name!r}; destination already exists"
                )
            shutil.move(str(item), str(target))
        nested.rmdir()
        logger.info("Un-nested chest_xray folder contents successfully.")

    def _validate_counts(self, destination: Path) -> dict:
        expected = self._expected_counts()
        actual = {}
        for split, labels in expected.items():
            split_dir = destination / split
            if not split_dir.is_dir():
                raise RuntimeError(f"missing split directory: {split_dir}")
            present = {entry.name for entry in split_dir.iterdir() if entry.is_dir()}
            if present != set(labels):
                raise RuntimeError(
                    f"{split} classes mismatch: expected {sorted(labels)}, "
                    f"found {sorted(present)}"
                )
            actual[split] = {}
            for label in labels:
                count = self._count_images(split_dir / label)
                actual[split][label] = count
        if actual != expected:
            raise RuntimeError(
                f"dataset image counts do not match config: expected {expected}, "
                f"found {actual}"
            )
        return actual

    def _swap_splits(self, staging: Path) -> None:
        expected = self._expected_counts()
        try:
            for split in expected:
                final = self.unzip_dir / split
                backup = self.unzip_dir / f".{split}.old"
                if backup.exists():
                    shutil.rmtree(backup)
                if final.exists():
                    os.replace(final, backup)
                os.replace(staging / split, final)
            for split in expected:
                backup = self.unzip_dir / f".{split}.old"
                if backup.exists():
                    shutil.rmtree(backup)
        except Exception:
            for split in expected:
                final = self.unzip_dir / split
                backup = self.unzip_dir / f".{split}.old"
                if backup.exists():
                    if final.exists():
                        shutil.rmtree(final)
                    os.replace(backup, final)
            raise

    def extract_zip_file(self):
        if self._dataset_is_complete():
            if not self.local_data_file.is_file() or self._archive_matches_manifest():
                logger.info("Prepared dataset already present; skipping extraction.")
                return
            if not self._zip_is_valid(self.local_data_file):
                logger.warning(
                    "Archive is corrupt; keeping the existing prepared dataset."
                )
                self.local_data_file.unlink()
                return

        if not self.local_data_file.is_file():
            raise FileNotFoundError(
                f"dataset archive not found: {self.local_data_file}"
            )
        if not self._zip_is_valid(self.local_data_file):
            raise RuntimeError(
                f"dataset archive is corrupt: {self.local_data_file}; "
                "delete it and run Download_file again"
            )
        self._verify_checksum(self.local_data_file)

        self.unzip_dir.mkdir(parents=True, exist_ok=True)
        staging = self.unzip_dir / ".extract_tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            logger.info("Extracting %s", self.local_data_file)
            self._safe_extract(self.local_data_file, staging)
            self._flatten_chest_xray(staging)
            macosx = staging / "__MACOSX"
            if macosx.exists():
                shutil.rmtree(macosx)
            counts = self._validate_counts(staging)

            archive_sha256 = self._sha256(self.local_data_file)
            manifest = {
                "dataset_slug": self.config.dataset_slug,
                "archive_sha256": archive_sha256,
                "archive_size": self.local_data_file.stat().st_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "counts": counts,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

            self._swap_splits(staging)
            os.replace(staging / "manifest.json", self.manifest_path)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        if not self.config.keep_zip:
            self.local_data_file.unlink(missing_ok=True)

        logger.info("Dataset prepared in %s with counts %s", self.unzip_dir, counts)
