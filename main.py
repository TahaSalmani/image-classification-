from cnnclassifier import logger
from cnnclassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipline
from cnnclassifier.pipeline.stage_02_prepare_base_model import PreparingBaseModelTrainingPipeline
from cnnclassifier.pipeline.stage_03_training import ModelTrainingPipeline
from cnnclassifier.pipeline.stage_04_Evaluation import EvaluationPipeline


def run_stage(stage_name, pipeline_cls):
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")
    pipeline_cls().main()
    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")


if __name__ == "__main__":
    stages = [
        ("Data Ingestion", DataIngestionTrainingPipline),
        ("Prepare Base Model", PreparingBaseModelTrainingPipeline),
        ("Training", ModelTrainingPipeline),
        ("Evaluation", EvaluationPipeline),
    ]
    for name, pipeline_cls in stages:
        try:
            run_stage(name, pipeline_cls)
        except Exception as e:
            logger.exception(e)
            raise e
