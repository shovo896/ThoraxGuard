"""Configuration entity definitions for the ThoraxGuard pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Resolved paths for the external dataset and ingestion artifacts."""

    root_dir: Path
    script_file: Path
    source_dir: Path
    manifest_file: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration needed to build and save the base CNN model."""

    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list[int]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration needed to train and save the classifier model."""

    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list[int]
    
