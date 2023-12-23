from thyroid_detector.entity import config_entity , artifact_entity
from thyroid_detector.logger import logging
from thyroid_detector.exception import ThyroidException
from thyroid_detector import utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from thyroid_detector.config import TARGET_COLUMN
import pandas as pd 
import numpy as np
import os, sys


class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
             self.data_transformation_config=data_transformation_config
             self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise ThyroidException(e, sys)   

    @classmethod
    def get_data_transformation_object(cls):
        try:
            standard_scaler = StandardScaler()
            return standard_scaler
        except Exception as e :
            raise ThyroidException(e, sys)             

    def initiate_data_transformation(self)-> artifact_entity.DataTransformationArtifact:   
        try:
            #reading train and test file path
            logging.info("Readind training and testing file")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

             #transforming features
            logging.info("Encoding the  features")
            label_encoder = LabelEncoder()
            for column in train_df.columns:
               train_df[column] = label_encoder.fit_transform(train_df[column])
               test_df[column] = label_encoder.fit_transform(test_df[column])

            #Selecting input feature for train and test dataset
            logging.info("Selecting input features for train and test dataframe")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)
            
            #Selecting target feature for train and test dataset
            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_arr = train_df[TARGET_COLUMN]
            target_feature_test_arr = test_df[TARGET_COLUMN]
            
           
            logging.info("Transforming input feature")
            transformation_pipeline = DataTransformation.get_data_transformation_object()
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input feature
            input_feature_train_arr=transformation_pipeline.fit_transform(input_feature_train_df)
            input_feature_test_arr=transformation_pipeline.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr , target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr , target_feature_test_arr]

            #Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path ,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path , 
                                        array=test_arr)                 

            utils.save_object(file_path=self.data_transformation_config.transform_object_path ,
                              obj=transformation_pipeline)  

            utils.save_object(file_path=self.data_transformation_config.label_encoder_path, 
                              obj=label_encoder)

                         

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                                   transform_object_path=self.data_transformation_config.transform_object_path,
                                   transformed_train_path=self.data_transformation_config.transformed_train_path,
                                   transformed_test_path=self.data_transformation_config.transformed_test_path,
                                   label_encoder_path=self.data_transformation_config.label_encoder_path
                                   ) 

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact                                                                                                                           

        except Exception as e:
            raise ThyroidException(e, sys)     