"""Configuration manager for the ThoraxGuard training workflow."""

from __future__ import annotations

from pathlib import Path

from cancer.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig
from cnnClassifier.utils.common import create_directories, read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"


class ConfigurationManager:
    """Read project YAML files and return resolved configuration entities."""

    def __init__(
        self,
        config_filepath: str | Path = CONFIG_FILE_PATH,
        params_filepath: str | Path = PARAMS_FILE_PATH,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.config_filepath = self._resolve_path(config_filepath)
        self.params_filepath = self._resolve_path(params_filepath)

        self.config = read_yaml(self.config_filepath)
        self.params = read_yaml(self.params_filepath)

        if not self.config.get("artifacts_root"):
            raise ValueError("config.yaml must define artifacts_root")

        self.artifacts_root = self._resolve_path(self.config.artifacts_root)
        create_directories([self.artifacts_root])

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        if not self.config.get("data_ingestion"):
            raise ValueError("config.yaml must define data_ingestion")

        config = self.config.data_ingestion
        required_keys = {"root_dir", "script_file", "source_dir", "manifest_file"}
        missing_keys = required_keys.difference(config.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"data_ingestion is missing required key(s): {missing}")

        root_dir = self._resolve_path(config.root_dir)
        script_file = self._resolve_path(config.script_file)
        source_dir = self._resolve_path(config.source_dir)
        manifest_file = self._resolve_path(config.manifest_file)

        if not root_dir.is_relative_to(self.artifacts_root):
            raise ValueError("data_ingestion.root_dir must be inside artifacts_root")
        if not script_file.is_relative_to(root_dir):
            raise ValueError("data_ingestion.script_file must be inside data_ingestion.root_dir")
        if not manifest_file.is_relative_to(root_dir):
            raise ValueError("data_ingestion.manifest_file must be inside data_ingestion.root_dir")
        if source_dir.is_relative_to(self.artifacts_root):
            raise ValueError("data_ingestion.source_dir must remain outside artifacts_root")
        if not script_file.is_file():
            raise FileNotFoundError(f"Data-ingestion script not found: {script_file}")
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {source_dir}")

        create_directories([root_dir])
        return DataIngestionConfig(
            root_dir=root_dir,
            script_file=script_file,
            source_dir=source_dir,
            manifest_file=manifest_file,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        if not self.config.get("prepare_base_model"):
            raise ValueError("config.yaml must define prepare_base_model")

        config = self.config.prepare_base_model
        required_keys = {"root_dir", "base_model_path", "updated_base_model_path"}
        missing_keys = required_keys.difference(config.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"prepare_base_model is missing required key(s): {missing}")

        root_dir = self._resolve_path(config.root_dir)
        base_model_path = self._resolve_path(config.base_model_path)
        updated_base_model_path = self._resolve_path(config.updated_base_model_path)

        create_directories([root_dir])
        return PrepareBaseModelConfig(
            root_dir=root_dir,
            base_model_path=base_model_path,
            updated_base_model_path=updated_base_model_path,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    def get_training_config(self) -> TrainingConfig:
        if not self.config.get("training"):
            raise ValueError("config.yaml must define training")

        config = self.config.training
        required_keys = {
            "root_dir",
            "trained_model_path",
            "updated_base_model_path",
            "training_data",
        }
        missing_keys = required_keys.difference(config.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"training is missing required key(s): {missing}")

        root_dir = self._resolve_path(config.root_dir)
        trained_model_path = self._resolve_path(config.trained_model_path)
        updated_base_model_path = self._resolve_path(config.updated_base_model_path)
        training_data = self._resolve_path(config.training_data)

        if not root_dir.is_relative_to(self.artifacts_root):
            raise ValueError("training.root_dir must be inside artifacts_root")
        if not trained_model_path.is_relative_to(root_dir):
            raise ValueError("training.trained_model_path must be inside training.root_dir")
        if not updated_base_model_path.is_file():
            raise FileNotFoundError(f"Updated base model not found: {updated_base_model_path}")
        if not training_data.is_dir():
            raise FileNotFoundError(f"Training data directory not found: {training_data}")

        create_directories([root_dir])
        return TrainingConfig(
            root_dir=root_dir,
            trained_model_path=trained_model_path,
            updated_base_model_path=updated_base_model_path,
            training_data=training_data,
            params_epochs=self.params.EPOCHS,
            params_batch_size=self.params.BATCH_SIZE,
            params_is_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
        )
