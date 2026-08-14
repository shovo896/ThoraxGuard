"""Configuration objects shared by the notebook and the ingestion script."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cnnClassifier.utils.common import create_directories, read_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataIngestionConfig:
    """Resolved paths for the external CT dataset and its artifact metadata."""

    root_dir: Path
    script_file: Path
    source_dir: Path
    manifest_file: Path


class ConfigurationManager:
    """Load ``config.yaml`` using the same data-ingestion schema as ``mara.ipynb``."""

    def __init__(self, config_filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_filepath = Path(config_filepath).expanduser().resolve()
        self.project_root = self.config_filepath.parent.parent
        self.config = read_yaml(self.config_filepath)

        if not self.config.get("artifacts_root"):
            raise ValueError("config.yaml must define artifacts_root")

        self.artifacts_root = self._resolve_path(self.config.artifacts_root)
        create_directories([self.artifacts_root])

    def _resolve_path(self, path_value: str | Path) -> Path:
        """Resolve config paths relative to the repository root when necessary."""
        path = Path(path_value).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Return validated paths used by ``mara.ipynb`` and data_ingestion.py.

        ``source_dir`` deliberately stays outside ``artifacts/``; only the
        ingestion script and its generated manifest are kept in artifacts.
        """
        if not self.config.get("data_ingestion"):
            raise ValueError("config.yaml must define data_ingestion")

        ingestion = self.config.data_ingestion
        required_keys = {"root_dir", "script_file", "source_dir", "manifest_file"}
        missing_keys = required_keys.difference(ingestion.keys())
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"data_ingestion is missing required key(s): {missing}")

        root_dir = self._resolve_path(ingestion.root_dir)
        script_file = self._resolve_path(ingestion.script_file)
        source_dir = self._resolve_path(ingestion.source_dir)
        manifest_file = self._resolve_path(ingestion.manifest_file)

        if not root_dir.is_relative_to(self.artifacts_root):
            raise ValueError("data_ingestion.root_dir must be inside artifacts_root")
        if not script_file.is_relative_to(root_dir):
            raise ValueError("data_ingestion.script_file must be inside data_ingestion.root_dir")
        if not manifest_file.is_relative_to(root_dir):
            raise ValueError("data_ingestion.manifest_file must be inside data_ingestion.root_dir")
        if not script_file.is_file():
            raise FileNotFoundError(f"Data-ingestion script not found: {script_file}")
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
        if source_dir.is_relative_to(self.artifacts_root):
            raise ValueError("data_ingestion.source_dir must remain outside artifacts_root")

        class_dirs = [path for path in source_dir.iterdir() if path.is_dir()]
        if len(class_dirs) < 2:
            raise ValueError(f"Expected at least two class folders in {source_dir}")

        create_directories([root_dir])
        return DataIngestionConfig(
            root_dir=root_dir,
            script_file=script_file,
            source_dir=source_dir,
            manifest_file=manifest_file,
        )
