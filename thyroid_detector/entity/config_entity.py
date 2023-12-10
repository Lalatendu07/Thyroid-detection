import os,sys
from datetime import datetime
from thyroid_detector.exception import ThyroidException

FILE_NAME = 'thyroid.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

class TrainingPipelineConfig:

    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}") 
        except Exception as e :
            raise ThyroidException(e , sys)

class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = "Thyroid_database"
            self.collection_name = "thyroid_data"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.features_store_file_path = os.path.join(self.data_ingestion_dir,'features_store',FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,'dataset',TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,'dataset',TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e :
            raise ThyroidException(e, sys)

class DataValidationConfig:...
class DataTransformationConfig:...
class ModelTrainerConfig:...
class ModelEvaluationConfig:...
class ModelPusherConfig:...