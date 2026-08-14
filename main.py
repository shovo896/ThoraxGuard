"""ThoraxGuard pipeline entry point."""

from __future__ import annotations

from cnnClassifier.pipeline.stage_1 import (
    STAGE_NAME as DATA_INGESTION_STAGE,
    DataIngestionArtifact,
    main as run_data_ingestion,
)
from cnnClassifier.utils import logger
from cancer.pipeline.stage_02_prepare_basemodel import (
    STAGE_NAME as PREPARE_BASE_MODEL_STAGE,
    PrepareBaseModelArtifact,
    main as run_prepare_base_model,
)


def main() -> tuple[DataIngestionArtifact, PrepareBaseModelArtifact]:
    """Run the configured ThoraxGuard pipeline stages."""
    try:
        logger.info(">>>>>> stage %s started <<<<<<", DATA_INGESTION_STAGE)
        data_artifact = run_data_ingestion()
        logger.info(">>>>>> stage %s completed <<<<<<", DATA_INGESTION_STAGE)

        logger.info(">>>>>> stage %s started <<<<<<", PREPARE_BASE_MODEL_STAGE)
        model_artifact = run_prepare_base_model()
        logger.info(">>>>>> stage %s completed <<<<<<", PREPARE_BASE_MODEL_STAGE)

        return data_artifact, model_artifact
    except Exception:
        logger.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
