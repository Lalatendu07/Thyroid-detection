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
    def get_data_transformation_object(cls)->Pipeline:
        try:
            standard_scaler = StandardScaler()
            constant_pipeline=Pipeline(steps=['StandardScaler',standard_scaler])
        except Exception as e :
            raise ThyroidException(e, sys)             

    def initiate_data_transformation(self)-> artifact_entity.DataTransformationArtifact:   
        try:
            #reading train and test file path
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            #Selecting input feature for train and test dataset
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #Selecting target feature for train and test dataset
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            #transformation on target column
            target_feature_train_arr=label_encoder.transform(target_feature_train_df)
            target_feature_test_arr=label_encoder.transform(target_feature_test_df)

            transformation_pipeline = DataTransformation.get_data_transformation_object()
            transformation_pipeline.fit(input_feature_train_df)

            #transforming input feature
            input_feature_train_arr=transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr=transformation_pipeline.transform(input_feature_test_df)


        except Exception as e:
            raise ThyroidException(e, sys)     