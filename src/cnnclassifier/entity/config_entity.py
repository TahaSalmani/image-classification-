from dataclasses import dataclass
from pathlib  import Path

@dataclass(frozen=True)
class DataIngestionConfig :
    root_dir: Path
    local_data_file : Path
    unzip_dir : Path
    dataset_slug : str
    expected_counts : dict
    download_timeout : int
    max_retries : int
    verify_sha256 : str
    keep_zip : bool
@dataclass (frozen=True)
class PrepareBaseModelConfig :
    root_dir : Path
    base_model_path  : Path
    updated_base_model_path : Path
    params_image_size : list
    params_learning_rate : float
    params_include_top : bool
    params_weight : str
    params_classes : int
    params_freeze_till : int

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path : Path
    checkpoint_model_path : Path
    training_data : Path
    params_epochs : int
    params_batch_size : int
    params_is_augmentation : bool
    params_image_size : list
    params_learning_rate : float
@dataclass(frozen=True)
class EvaluationConfig :
    path_of_model :Path
    test_data : Path
    params_image_size : list
    params_batch_size : int
    all_params : dict