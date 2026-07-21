from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from cnnclassifier import  logger

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

