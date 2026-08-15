"""Stage 4: evaluate the trained classifier model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cancer.components.model_evaluation import ModelEvaluation
from config.configuration import ConfigurationManager
from cnnClassifier.utils import logger


STAGE_NAME = "Model Evaluation"


@dataclass(frozen=True)
class ModelEvaluationArtifact:
    scores_file: Path
    loss: float
    accuracy: float


class ModelEvaluationPipeline:
    def __init__(self, evaluator: ModelEvaluation | None = None) -> None:
        if evaluator is None:
            config = ConfigurationManager().get_evaluation_config()
            evaluator = ModelEvaluation(config=config)
        self.evaluator = evaluator

    def run(self, log_to_mlflow: bool = False) -> ModelEvaluationArtifact:
        logger.info("%s started", STAGE_NAME)

        if log_to_mlflow:
            self.evaluator.log_into_mlflow()
        else:
            self.evaluator.evaluation()

        artifact = ModelEvaluationArtifact(
            scores_file=self.evaluator.config.scores_file,
            loss=self.evaluator.score["loss"],
            accuracy=self.evaluator.score["accuracy"],
        )
        logger.info(
            "%s completed: scores=%s, loss=%s, accuracy=%s",
            STAGE_NAME,
            artifact.scores_file,
            artifact.loss,
            artifact.accuracy,
        )
        return artifact

    def main(self) -> ModelEvaluationArtifact:
        return self.run()


def main(log_to_mlflow: bool = False) -> ModelEvaluationArtifact:
    config = ConfigurationManager().get_evaluation_config()
    evaluator = ModelEvaluation(config=config)
    return ModelEvaluationPipeline(evaluator=evaluator).run(log_to_mlflow=log_to_mlflow)


if __name__ == "__main__":
    main()
