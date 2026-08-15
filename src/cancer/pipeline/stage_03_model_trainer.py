"""Stage 3: train the classifier model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.configuration import ConfigurationManager
from cancer.components.model_trainer import ModelTrainer
from cnnClassifier.utils import logger


STAGE_NAME = "Training"


@dataclass(frozen=True)
class ModelTrainingArtifact:
    trained_model_path: Path


class ModelTrainingPipeline:
    def __init__(self, trainer: ModelTrainer | None = None) -> None:
        if trainer is None:
            config = ConfigurationManager().get_training_config()
            trainer = ModelTrainer(config=config)
        self.trainer = trainer

    def run(self) -> ModelTrainingArtifact:
        logger.info("%s started", STAGE_NAME)
        self.trainer.get_base_model()
        self.trainer.train_valid_generator()
        self.trainer.train()

        artifact = ModelTrainingArtifact(
            trained_model_path=self.trainer.config.trained_model_path
        )
        logger.info("%s completed: trained_model=%s", STAGE_NAME, artifact.trained_model_path)
        return artifact

    def main(self) -> ModelTrainingArtifact:
        return self.run()


def main() -> ModelTrainingArtifact:
    config = ConfigurationManager().get_training_config()
    trainer = ModelTrainer(config=config)
    return ModelTrainingPipeline(trainer=trainer).run()


if __name__ == "__main__":
    main()
