"""Stage 2: prepare the VGG16 base model and updated classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.configuration import ConfigurationManager
from cancer.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.utils import logger


STAGE_NAME = "Prepare Base Model"


@dataclass(frozen=True)
class PrepareBaseModelArtifact:
    base_model_path: Path
    updated_base_model_path: Path


class PrepareBaseModelStage:
    def __init__(self, component: PrepareBaseModel) -> None:
        self.component = component

    def run(self) -> PrepareBaseModelArtifact:
        logger.info("%s started", STAGE_NAME)
        self.component.get_base_model()
        self.component.update_base_model()

        artifact = PrepareBaseModelArtifact(
            base_model_path=self.component.config.base_model_path,
            updated_base_model_path=self.component.config.updated_base_model_path,
        )
        logger.info(
            "%s completed: base_model=%s, updated_base_model=%s",
            STAGE_NAME,
            artifact.base_model_path,
            artifact.updated_base_model_path,
        )
        return artifact


def main() -> PrepareBaseModelArtifact:
    config = ConfigurationManager().get_prepare_base_model_config()
    component = PrepareBaseModel(config=config)
    return PrepareBaseModelStage(component=component).run()


if __name__ == "__main__":
    main()
