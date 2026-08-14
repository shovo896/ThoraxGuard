from cnnClassifier.config.configuration import ConfigurationManager 
from cnnClassifier.component.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger 

STAGE_NAME = "Prepare base model" 

class PrepareBaseModelStage:
    def __init__(self, config: ConfigurationManager):
        self.config = config 
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model=PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        
        
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        obj=PrepareBaseModelStage()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx") 
        
    except Exception as e:
        logger.exception(e)
        raise e