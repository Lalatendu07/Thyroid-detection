from thyroid_detector.entity import config_entity , artifact_entity
from thyroid_detector.logger import logging
from thyroid_detector.exception import ThyroidException
from thyroid_detector import utils
from scipy.stats import ks_2samp
import pandas as pd 
import numpy as np
import os, sys


class DataValidation:

    def __init__(self,data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} DATA VALIDATION {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e :
            raise ThyroidException(e, sys)


    def drop_missing_value_column(self,df:pd.DataFrame,report_key_name:str)->pd.DataFrame:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            logging.info(f"Selecting column names which null values above {threshold}")
            #Selecting column names which null values
            drop_column_names = null_report[null_report>threshold].index
            logging.info(f"Columns to drop {drop_column_names}")
            self.validation_error[report_key_name]=drop_column_names
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #Return None if ni column left
            if len(df.columns)==0:
                return None
            return df    
        except Exception as e:
            raise ThyroidException(e, sys)     


    def is_required_column_exist(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True            
        except Exception as e:
            raise ThyroidException(e , sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data=base_df[base_column],current_df[base_column]
                #Null hypotesis is that both columns are from same distribution
                same_distribution = ks_2samp(base_data,current_data)

                if same_distribution.pvalue>0.05:
                    drift_report[base_column]={
                        'pvalue':same_distribution.pvalue,
                        'same_distribution':True
                    }
                else:
                    drift_report[base_column]={
                        'pvalue':same_distribution.pvalue,
                        'same_distribution':False
                    }

            self.validation_error[report_key_name]=drift_report

        except Exception as e :
            raise ThyroidException(e, sys)       

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info('Reading base dataframe')
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({'?':np.NAN})
            logging.info("Drop null column in base dataframe")
            base_df=self.drop_missing_value_column(df=base_df,report_key_name='missing_values_within_base_dataset')

            logging.info('Reading train dataframe')
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info('Reading test dataframe')
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
             
            logging.info("Drop null column in train dataframe")
            train_df=self.drop_missing_value_column(df=train_df, report_key_name='missing_columns_within_train_dataset')
            logging.info("Drop null column in test dataframe")
            test_df=self.drop_missing_value_column(df=test_df, report_key_name='missing_columns_within_test_dataset')

            logging.info("Is all required columns present train dataframe")
            train_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=train_df,report_key_name='data_drift_within_train_dataset')
            logging.info("Is all required columns present test dataframe")
            test_df_column_status = self.is_required_column_exist(base_df=base_df, current_df=test_df,report_key_name='data_drift_within_train_dataset')

            if train_df_column_status:
                logging.info('Detecting data drift in train dataset')
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name='Data_drift_train_dataset')
            if test_df_column_status:
                logging.info('Detecting data drift in test dataset')
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name='Data_drift_test_dataset')

            #Write the report
            logging.info('Write the report in yaml file')
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f'Data Validation artifact : {data_validation_artifact}')
            return data_validation_artifact    
                

        except Exception as e :
            raise ThyroidException(e, sys)         