from pathlib import Path

from cnnclassifier.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
)
from cnnclassifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=Path("config/config.yaml"),
        params_filepath=Path("params.yaml"),
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            dataset_slug=config.dataset_slug,
            expected_counts={
                split: {label: int(count) for label, count in labels.items()}
                for split, labels in config.expected_counts.items()
            },
            download_timeout=int(config.download_timeout),
            max_retries=int(config.max_retries),
            verify_sha256=config.verify_sha256 or "",
            keep_zip=bool(config.keep_zip),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([Path(config.root_dir)])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES,
            params_weight=self.params.WEIGHT,
            params_include_top=self.params.INCLUDE_TOP,
            params_freeze_till=int(self.params.FREEZE_TILL),
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        create_directories(
            [
                Path(config.root_dir),
                Path(config.tensorboard_root_log_dir),
                Path(config.checkpoint_model_filepath).parent,
            ]
        )
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        prepare_callbacks = self.config.prepare_callbacks
        training_data = Path(self.config.data_ingestion.unzip_dir) / "train"
        create_directories(
            [
                Path(training.root_dir),
                Path(prepare_callbacks.checkpoint_model_filepath).parent,
            ]
        )

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.training_model_path),
            updated_base_model_path=Path(
                prepare_base_model.updated_base_model_path
            ),
            checkpoint_model_path=Path(
                prepare_callbacks.checkpoint_model_filepath
            ),
            training_data=training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        test_data = Path(self.config.data_ingestion.unzip_dir) / "test"
        return EvaluationConfig(
            path_of_model=Path(self.config.training.training_model_path),
            test_data=test_data,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
