from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from cnnclassifier import  logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from  cnnclassifier.pipeline.stage_02_prepare_base_model import  PreparingBaseModelTrainingPipeline
from  cnnclassifier.pipeline.stage_03_training import ModelTrainingPipeline
from  cnnclassifier.pipeline.stage_04_Evaluation import EvaluationPipeline
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


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")

    except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Training"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e






STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise e

