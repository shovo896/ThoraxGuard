"""Stage 1: scan the external CT dataset and create its artifact manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cnnClassifier.config.configuration import ConfigurationManager, DataIngestionConfig
from cnnClassifier.utils import logger


STAGE_NAME = "Data Ingestion"


@dataclass(frozen=True)
class DataIngestionArtifact:
    """Metadata produced after scanning the same dataset used by ``mara.ipynb``."""

    source_dir: Path
    manifest_file: Path
    class_counts: dict[str, int]
    total_images: int


class DataIngestionStage:
    """Run the configured artifact script without copying the external dataset."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def run(self) -> DataIngestionArtifact:
        """Generate and validate the manifest for ``config.source_dir``."""
        logger.info("%s started; dataset remains at %s", STAGE_NAME, self.config.source_dir)
        subprocess.run(
            [sys.executable, str(self.config.script_file)],
            cwd=self.config.root_dir.parent,
            check=True,
        )

        with self.config.manifest_file.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        manifest_source = Path(manifest["source_dir"]).expanduser().resolve()
        if manifest_source != self.config.source_dir:
            raise ValueError(
                "Manifest source does not match config source: "
                f"{manifest_source} != {self.config.source_dir}"
            )

        class_counts = manifest["class_counts"]
        total_images = manifest["total_images"]
        if not isinstance(class_counts, dict) or total_images != sum(class_counts.values()):
            raise ValueError("Manifest contains inconsistent class counts")

        artifact = DataIngestionArtifact(
            source_dir=manifest_source,
            manifest_file=self.config.manifest_file,
            class_counts=class_counts,
            total_images=total_images,
        )
        logger.info(
            "%s completed: %s images across %s",
            STAGE_NAME,
            artifact.total_images,
            sorted(artifact.class_counts),
        )
        return artifact


def main() -> DataIngestionArtifact:
    """Run Stage 1 using the config schema already used by ``mara.ipynb``."""
    config = ConfigurationManager().get_data_ingestion_config()
    return DataIngestionStage(config).run()


if __name__ == "__main__":
    main()
