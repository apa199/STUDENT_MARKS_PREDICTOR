from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from scr.exception import CustomException
from scr.utils import save_object, evaluate_models
from sklearn.metrics import r2_score

import os
import sys
@dataclass
class ModelTrainerConfig():
    model_obj_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array,test_array):
        try:
            X_train,Y_train,x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "RandomForest":RandomForestRegressor(),
                "LinearRegression":LinearRegression(),
                "DecisionTree":DecisionTreeRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                }}

            model_report:dict=evaluate_models(X_train=X_train,y_train=Y_train,X_test=x_test,y_test=y_test,
                                             models=models)
            

            Best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(Best_model_score)]
            best_model=models[best_model_name]

            save_object(file_path=self.model_trainer_config.model_obj_path,
                        obj=best_model)
            predicted=best_model.predict(x_test)

            r2 = r2_score(y_test, predicted)
            return r2                      
        except Exception as e:
            raise CustomException(e,sys)