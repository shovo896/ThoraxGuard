"""ThoraxGuard pipeline entry point."""

from __future__ import annotations

from cancer.pipeline.stage_02_prepare_basemodel import (
    STAGE_NAME as PREPARE_BASE_MODEL_STAGE,
    PrepareBaseModelArtifact,
    main as run_prepare_base_model,
)
from cancer.pipeline.stage_03_model_trainer import (
    STAGE_NAME as TRAINING_STAGE,
    ModelTrainingArtifact,
    main as run_model_training,
)
from cnnClassifier.pipeline.stage_1 import (
    STAGE_NAME as DATA_INGESTION_STAGE,
    DataIngestionArtifact,
    main as run_data_ingestion,
)
from cnnClassifier.utils import logger


def _run_stage(stage_name: str, stage_callable):
    logger.info(">>>>>> stage %s started <<<<<<", stage_name)
    artifact = stage_callable()
    logger.info(">>>>>> stage %s completed <<<<<<", stage_name)
    return artifact


def main() -> tuple[DataIngestionArtifact, PrepareBaseModelArtifact, ModelTrainingArtifact]:
    """Run the configured ThoraxGuard stages in order."""
    try:
        data_artifact = _run_stage(DATA_INGESTION_STAGE, run_data_ingestion)
        model_artifact = _run_stage(PREPARE_BASE_MODEL_STAGE, run_prepare_base_model)
        training_artifact = _run_stage(TRAINING_STAGE, run_model_training)
        return data_artifact, model_artifact, training_artifact
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
    

