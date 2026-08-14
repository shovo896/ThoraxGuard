"""ThoraxGuard pipeline entry point."""

from __future__ import annotations

from cnnClassifier.pipeline.stage_1 import STAGE_NAME, DataIngestionArtifact, main as run_data_ingestion
from cnnClassifier.utils import logger



def main() -> DataIngestionArtifact:
    """Run Stage 1 and return metadata for the external CT dataset."""
    try:
        logger.info("%s pipeline started", STAGE_NAME)
        artifact = run_data_ingestion()
        logger.info(
            "%s pipeline completed: %s images; class counts=%s",
            STAGE_NAME,
            artifact.total_images,
            artifact.class_counts,
        )
        return artifact
    except Exception:
        logger.exception("%s pipeline failed", STAGE_NAME)
        
STAGE_NAME = "Prepare base model"
 try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        obj=PrepareBaseModelStage()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx") 
        
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    main()
