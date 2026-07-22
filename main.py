from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from cnnclassifier import  logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from  cnnclassifier.pipeline.stage_02_prepare_base_model import  PreparingBaseModelTrainingPipeline


STAGE_NAME = "Data Ingestion Stage "
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

    except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "prepare base model "

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PreparingBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

    except Exception as e:
        logger.exception(e)
        raise e



