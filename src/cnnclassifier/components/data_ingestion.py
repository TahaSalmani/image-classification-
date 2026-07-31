import os
import urllib.request as request
import zipfile
import shutil
from pathlib import Path
from cnnclassifier import logger
from cnnclassifier.utils.common import get_size
from cnnclassifier.entity.config_entity import DataIngestionConfig
from cnnclassifier.constants import *


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def Download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading dataset from Kaggle...")

            dataset_slug = "paultimothymooney/chest-xray-pneumonia"
            download_dir = os.path.dirname(self.config.local_data_file)
            os.makedirs(download_dir, exist_ok=True)

            os.system(f"kaggle datasets download -d {dataset_slug} -p {download_dir}")

            downloaded_zip = os.path.join(
                download_dir, "chest-xray-pneumonia.zip"
            )
            if os.path.exists(downloaded_zip):
                os.rename(downloaded_zip, self.config.local_data_file)

            logger.info(f"Dataset downloaded to {self.config.local_data_file}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file) as zip_ref:
            zip_ref.extractall(unzip_path)

        macosx_path = os.path.join(unzip_path, "__MACOSX")
        if os.path.exists(macosx_path):
            shutil.rmtree(macosx_path)
            logger.info("Removed __MACOSX folder successfully.")

        nested_chest_xray = os.path.join(unzip_path, "chest_xray")
        if os.path.exists(nested_chest_xray):
            for item in os.listdir(nested_chest_xray):
                src = os.path.join(nested_chest_xray, item)
                dst = os.path.join(unzip_path, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

            shutil.rmtree(nested_chest_xray)
            logger.info("Un-nested chest_xray folder contents successfully.")