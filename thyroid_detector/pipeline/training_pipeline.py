from thyroid_detector.utils import get_collection_as_dataframe
from thyroid_detector.exception import ThyroidException
from thyroid_detector.entity import config_entity , artifact_entity
from thyroid_detector.components.data_ingestion import DataIngestion 
from thyroid_detector.components.data_validation import DataValidation
from thyroid_detector.components.data_transformation import DataTransformation
from thyroid_detector.components.model_trainer import ModelTrainer
from thyroid_detector.components.model_evaluation import ModelEvaluation
from thyroid_detector.components.model_pusher import ModelPusher
import os, sys


def start_training_pipeline():
    try:
        #Data Ingestion
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        #Data Validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                                         data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation() 

        #Data Transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()      

        #Model Trainer 
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        #Model Evaluation
        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_eval_config=model_eval_config, 
                                           data_ingestion_artifact=data_ingestion_artifact,
                                           data_transformation_artifact=data_transformation_artifact,
                                           model_trainer_artifact=model_trainer_artifact)
        model_eval_artifact = model_evaluation.initiate_model_evaluation()

        #Model Pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()  

    except Exception as e:
        raise ThyroidException(e, sys)