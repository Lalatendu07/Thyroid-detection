from thyroid_detector.predictor import ModelResolver
from thyroid_detector.entity import config_entity,artifact_entity
from thyroid_detector.exception import ThyroidException
from thyroid_detector.logger import logging
from thyroid_detector.utils import load_object
from sklearn.metrics import f1_score
from thyroid_detector.config import TARGET_COLUMN
import pandas as pd
import os,sys


class ModelEvaluation:

    def __init__(self,model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact, 
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"{'>>'*20} MODEL EVALUATION {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact 
            self.model_resolver=ModelResolver()
        except Exception as e:
            raise ThyroidException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            #If saved model folder has model then we will compare
            logging.info("if saved model folder has model then we will compare") 
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                      improved_accuracy=None)
                return model_eval_artifact 

            #Finding location of transformer , model and label encoder
            logging.info("Finding location transformer , model and label encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            label_encoder_path = self.model_resolver.get_latest_label_encoder_path()    

            #Previous trained objects
            logging.info("Previous trained objects of transformer , model and label encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            label_encoder = load_object(file_path=label_encoder_path)

            #Currently trained objects
            logging.info("Currently trained objects of transformer , model and label encoder")
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_label_encoder = load_object(file_path=self.data_transformation_artifact.label_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path )
            test_df_cr=test_df.copy()

            #accuracy using previous trained model 
            logging.info("accuracy using previous trained model")
            for column in test_df.columns:
                test_df[column] = label_encoder.fit_transform(test_df[column])
                
            input_arr = transformer.transform(test_df.iloc[:,:-1])
            y_pred = model.predict(input_arr)
            y_true = test_df[TARGET_COLUMN]
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')  
            logging.info(f"accuracy using previous trained model: {previous_model_score}") 

            #accuracy using current trained model 
            logging.info("accuracy using current trained model")
            for column in test_df_cr.columns:
                test_df_cr[column] = current_label_encoder.fit_transform(test_df_cr[column])

            input_arr = current_transformer.transform(test_df_cr.iloc[:,:-1])
            y_pred = current_model.predict(input_arr)
            y_true = test_df_cr[TARGET_COLUMN]    
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')  
            logging.info(f"accuracy using current trained model: {current_model_score}")  

            if current_model_score<previous_model_score:
                logging.info("Current trained model is not better than previous trained model")
                raise Exception("Current trained model is not better than previous trained model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
            improved_accuracy=current_model_score-previous_model_score)
            improved_accuracy=current_model_score-previous_model_score
            logging.info(f"model eval artifact: {improved_accuracy}")
            return model     

                                  
        except Exception as e:
            raise ThyroidException(e, sys)        