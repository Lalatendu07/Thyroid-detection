import pandas as pd
import numpy as np
from thyroid_detector.config import mongo_client
from thyroid_detector.logger import logging
from thyroid_detector.exception import ThyroidException
import os,sys
import yaml
import dill

def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    
    try:
        logging.info(f"Reading data from database: {database_name} and {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info("Found dataset")
        if "_id" in df.columns:
            logging.info("Dropping column: _id")
            df=df.drop("_id",axis=1)
        logging.info(f"Rows and columns in df: {df.shape}")
        return df    

    except Exception as e:
        raise ThyroidException(e, sys) from e  


 
