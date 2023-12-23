from thyroid_detector.exception import ThyroidException
from thyroid_detector.logger import logging
from thyroid_detector.predictor import ModelResolver
from thyroid_detector.utils import load_object
from datetime import datetime
import numpy as np
import pandas as pd
import os,sys

PREDICTION_DIR="prediction"


def batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
    except Exception as e:
        raise ThyroidException(e, sys)
