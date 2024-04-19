import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scr.utils import save_object
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scr.exception import CustomException
from scr.logger import logging

@dataclass
class DataTrasnformationConfig:
    preprocessor_obj_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTrasnformation:
    def __init__(self):
        self.data_trasnformation_config =DataTrasnformationConfig()


    def get_data_transformation_objects(self):
        try:
            numerical_features=['reading_score', 'writing_score']
            categorical_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline= Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ])
            logging.info('Numerical encoding done.:{numerical_features}')
            logging.info('Categorical encoding done.:{categorical_features}')

            preprocessor=ColumnTransformer(
                [
               ("num_pipeline",num_pipeline,numerical_features),
               ("categorical_pipeline",cat_pipeline,categorical_features)
            ]
            )
            
            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)  
        
        
    def initate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            


            logging.info('Read train and test data.')
            preprocessor_obj=self.get_data_transformation_objects()
            # preprocessor.fit(train_df)
            target_columns="math_score"
            
            input_feature_train_df=train_df.drop(columns=[target_columns], axis=1)
            target_feature_train_df=train_df[target_columns]
            

            input_feature_test_df=test_df.drop(columns=[target_columns], axis=1)
            target_feature_test_df=test_df[target_columns]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_trasnformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_trasnformation_config.preprocessor_obj_path
            )

            
        except Exception as e:
            raise CustomException(e,sys)