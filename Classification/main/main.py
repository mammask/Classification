"""
 script name: main.py
     purpose: to analyze HR data and predict whether an employee resigns or not
      author: mammaskon@gmail.com
"""

# Load packages
# ---------------------------------------------------------------------------------------------------------------------
import pandas as pd
from src import Manipulator
from sklearn.model_selection import train_test_split
from src import Cross_Validator
from src import Configurator as conf
import numpy as np

# Input parameters
# ---------------------------------------------------------------------------------------------------------------------
data_path_file = eval(conf.ConfigClassification().get('data','data_path', 2))
target_var = eval(conf.ConfigClassification().get('variable type','target_variable', 2))
binary_var = eval(conf.ConfigClassification().get('variable type','binary_variable', 2))
train_size = eval(conf.ConfigClassification().get('cross validation','train_size', 2))
sampling_val = eval(conf.ConfigClassification().get('cross validation','sampling_val', 2))
classifer_val = eval(conf.ConfigClassification().get('classifiers','classifiers_list', 2))
grid_id = eval(conf.ConfigClassification().get('grid parameters','grid_id', 2))
nfold_id = eval(conf.ConfigClassification().get('cross validation','n_folds', 2))


# Load data
# ---------------------------------------------------------------------------------------------------------------------
input_data_origin = pd.read_csv(data_path_file, sep = ",")

# Feature engineering - Convert Character Variables to Binary Integer
# ------------------------------------
input_data_updated = input_data_origin
for feature_id in binary_var:
    print("Currently processing " + feature_id)
    manipulation_object = Manipulator.Data_Manipulator(data_object = input_data_updated,
                                                       variable_name = feature_id
                                                       )
    input_data_updated = manipulation_object.map_level()

# Randomly split the data
# ---------------------------------------------------------------------------------------------------------------------
train_data, test_data = train_test_split(input_data_updated, test_size=train_size, random_state=0)


# Perform K-Fold Cross Validation by performing sampling techniques to correct the class imbalance
# ---------------------------------------------------------------------------------------------------------------------
cv_results = Cross_Validator.CrossValidator(training_set=train_data,
                                            test_set=test_data,
                                            target_variable=target_var,
                                            kf_list = None)
# Define the number of partitions
# ---------------------------------------------------------------------------------------------------------------------
cv_results.cv_split(num_partitions=nfold_id, stratify_cv=True)

# Run Cross Validation and model fitting using random forest
# ---------------------------------------------------------------------------------------------------------------------
output_df, model_object_output = cv_results.parameter_iterator(grid_id=grid_id,
                                                               sampling_technique=sampling_val,
                                                               classifier=classifer_val)
