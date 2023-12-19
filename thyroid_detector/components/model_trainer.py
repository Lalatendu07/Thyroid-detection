from thyroid_detector.entity import config_entity , artifact_entity
from thyroid_detector.logger import logging
from thyroid_detector.exception import ThyroidException
from thyroid_detector import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
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

    def train_model(self,x,y):
        try:
            Dtree_clf = DecisionTreeClassifier()
            Dtree_clf.fit(x,y)
            return Dtree_clf
        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_model_trainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            
            logging.info("Splitting input and target feature from both train and test array.")
            x_train, y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info("Train the model")
            model = self.train_model(x=x_train, y=y_train)

            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train , y_pred=yhat_train,average='micro')

            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test , y_pred=yhat_test,average='micro')

            logging.info(f"Train score: {f1_train_score} and Test score: {f1_test_score}")
            #Check for overfitting , underfitting or expected score
            logging.info("Checking if model is underfitting or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                    expected accuracy: {self.model_trainer_config.expected_score}: model actual accuracy: {f1_test_score}")

            logging.info("Checking if model is overfitting or not")
            diff = abs(f1_train_score-f1_test_score)

            if diff>self.model_trainer_config.overfitting_thres:
                raise Exception(f"Train and test diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_thres}")

            #save the trained model
            logging.info("Saving the model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)   

            #prepare artifact
            logging.info("Prepare the artifact") 
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                                      model_path=self.model_trainer_config.model_path,
                                      f1_train_score=f1_train_score,
                                      f1_test_score=f1_test_score)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact                           
        except Exception as e :
            raise ThyroidException(e, sys)            