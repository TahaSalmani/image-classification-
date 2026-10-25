import random
import shutil
from pathlib import Path

import pandas as pd
import tensorflow as tf

from cnnclassifier.entity.config_entity import TrainingConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PREPROCESS = tf.keras.applications.vgg16.preprocess_input


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.class_weight = None

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False,
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    @staticmethod
    def _patient_id(image_path: Path) -> str:
        stem = image_path.stem
        if "_" in stem:
            return stem.split("_")[0]
        # NORMAL files use IDs like IM-0115-0001 or NORMAL2-IM-1427-0001;
        # the trailing segment is the image number, not the patient.
        parts = stem.split("-")
        return "-".join(parts[:-1]) if len(parts) > 1 else stem

    def _patient_split(self):
        root = Path(self.config.training_data)
        records = []
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for image_path in class_dir.rglob("*"):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_SUFFIXES
                ):
                    records.append(
                        {
                            "filename": str(image_path),
                            "class": class_dir.name,
                            "patient": self._patient_id(image_path),
                        }
                    )
        if not records:
            raise RuntimeError(f"no images found under {root}")

        frame = pd.DataFrame(records)
        rng = random.Random(42)
        val_patients = set()
        for label in sorted(frame["class"].unique()):
            patients = sorted(
                frame.loc[frame["class"] == label, "patient"].unique()
            )
            rng.shuffle(patients)
            count = max(1, round(len(patients) * 0.2))
            val_patients.update(patients[:count])

        train_frame = frame[~frame["patient"].isin(val_patients)].reset_index(
            drop=True
        )
        val_frame = frame[frame["patient"].isin(val_patients)].reset_index(
            drop=True
        )
        if train_frame["class"].nunique() < 2 or val_frame["class"].nunique() < 2:
            raise RuntimeError(
                "patient-grouped split produced a split without both classes"
            )
        return train_frame, val_frame

    def train_valid_generator(self):
        train_frame, val_frame = self._patient_split()
        target_size = tuple(self.config.params_image_size[:-1])
        dataflow_kwargs = dict(
            target_size=target_size,
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            interpolation="bilinear",
        )

        valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=PREPROCESS
        )
        self.valid_generator = valid_generator.flow_from_dataframe(
            dataframe=val_frame,
            x_col="filename",
            y_col="class",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=PREPROCESS,
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            )
        else:
            train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=PREPROCESS
            )

        self.train_generator = train_generator.flow_from_dataframe(
            dataframe=train_frame,
            x_col="filename",
            y_col="class",
            shuffle=True,
            seed=42,
            **dataflow_kwargs,
        )

        counts = train_frame["class"].value_counts().to_dict()
        total = sum(counts.values())
        class_count = len(counts)
        self.class_weight = {
            self.train_generator.class_indices[label]: total
            / (class_count * count)
            for label, count in counts.items()
        }

    @staticmethod
    def save_model(path, model: tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        self.steps_per_epoch = max(
            1, self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = max(
            1, self.valid_generator.samples // self.valid_generator.batch_size
        )

        checkpoint = Path(self.config.checkpoint_model_path)
        checkpoint.unlink(missing_ok=True)

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
            class_weight=self.class_weight,
        )

        # ModelCheckpoint keeps the best val_accuracy weights; publish that one
        # instead of the weights from the final (possibly overfitted) epoch.
        if checkpoint.is_file():
            shutil.copy2(checkpoint, self.config.trained_model_path)
        else:
            self.save_model(path=self.config.trained_model_path, model=self.model)
