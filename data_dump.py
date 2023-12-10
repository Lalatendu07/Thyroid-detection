import pymongo
import pandas as pd
import json
from thyroid_detector.config import mongo_client

# Provide the mongodb localhost url to connect python to mongodb.
client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATA_FILE_PATH = "/config/workspace/hypothyroid.csv"

DATABASE_NAME = "Thyroid_database"

COLLECTION_NAME = "thyroid_data"

if __name__ == "__main__":

    df = pd.read_csv(DATA_FILE_PATH)
    print(f'Rows and Column : {df.shape}')

    # Convert dataframe to json format
    df.reset_index(drop=True , inplace=True)
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    # Insert the converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)