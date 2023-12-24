from thyroid_detector.exception import ThyroidException
from thyroid_detector.logger import logging
from thyroid_detector.predictor import ModelResolver
from thyroid_detector.utils import load_object
from datetime import datetime
import numpy as np
import pandas as pd
import os,sys

PREDICTION_DIR="prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        logging.info("Replacing '?' witn np.NAN")
        df.replace(to_replace='?',value=np.NAN,inplace=True) 
        logging.info("filling the null values")
        df['sex'].fillna(str(df['sex'].mode()),inplace=True)
        df['age'].fillna(int(df['age'].median()),inplace=True)
        df['TSH'].fillna(int(df['TSH'].median()),inplace=True)
        df['T3'].fillna(int(df['T3'].median()),inplace=True)
        df['TT4'].fillna(int(df['TT4'].median()),inplace=True)
        df['T4U'].fillna(int(df['T4U'].median()),inplace=True)
        df['FTI'].fillna(int(df['FTI'].median()),inplace=True) 
        df.drop('TBG',axis=1,inplace=True)

        for col in df.columns:
            df[col]=df[col].astype('str')

        logging.info(f"Loading label encoder to encode dataset")
        label_encoder = load_object(file_path=model_resolver.get_latest_label_encoder_path())
        for column in df.columns:
            df[column] = label_encoder.fit_transform(df[column])

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
        input_arr = transformer.transform(df.iloc[:,:-1])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        prediction = prediction.astype('int')
        cat_prediction = label_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".data",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        return prediction_file_path    

    except Exception as e:
        raise ThyroidException(e, sys)
