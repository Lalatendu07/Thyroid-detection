from thyroid_detector.entity import config_entity , artifact_entity
from thyroid_detector.logger import logging
from thyroid_detector.exception import ThyroidException
from thyroid_detector import utils
from sklearn.tree import DecisionTreeClassifier
import os, sys


class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} MODEL TRAINING {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise ThyroidException(e, sys)  

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            pass
        except Exception as e :
            raise ThyroidException(e, sys)            