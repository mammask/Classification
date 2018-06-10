import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
import itertools as it
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class CrossValidator:

    def __init__(self,
                 training_set,
                 test_set,
                 target_variable,
                 kf_list
                 ):

        """
        :param training_set         : training dataset
        :param target_variable      : name of target variable
        :param kf_list              : list of k-fold vectors
        :param grid_id              : grid parameters
        """
        self.train_data = training_set
        self.test_data = test_set
        self.target_variable = target_variable
        self.kf_list = kf_list
        self.grid_df = None
        self.cv_score = None

    def cv_split(self, num_partitions, stratify_cv):

        if stratify_cv == True:
            self.kf_list = KFold(n_splits= num_partitions, shuffle=False, random_state = 0)
        else:
            self.kf_list = StratifiedKFold(n_splits=num_partitions, shuffle=False, random_state=0)

        return self.kf_list

    def cv_executor(self, sampling_technique, classifier, param_grid):

        # Perform cross validation
        accuracy = []
        for i, (cv_index_train, cv_index_test) in enumerate(self.kf_list.split(self.train_data.index,
                                                            self.train_data[self.target_variable])):

            # Obtain predictors and response variable for the k-1 training set
            X_res = self.train_data.ix[self.train_data.index[cv_index_train]].iloc[:, ~self.train_data.columns.isin(self.target_variable)]
            y_res = self.train_data.ix[self.train_data.index[cv_index_train]][self.target_variable]

            # Perform Sampling
            if sampling_technique == "oversample":
                print("Performing Over-sample within cross validation - step:" + str(i+1))
                # Perform Over-Sampling
                ros = RandomOverSampler(random_state=0)
                X_res, y_res = ros.fit_sample(X_res, y_res.values.ravel())

            if sampling_technique == "undersample":
                print("Performing Under-sample within cross validation - step:" + str(i+1))
                # Perform Over-Sampling
                ros = RandomUnderSampler(random_state=0)
                X_res, y_res = ros.fit_sample(X_res, y_res.values.ravel())

            # Fit Random Forest Classifier
            if classifier == "rf":
                print("Training Random Forest within cross validation - step:" + str(i + 1))
                clf = RandomForestClassifier(random_state=0, n_estimators=100)
                grid_search = GridSearchCV(clf, param_grid= param_grid)
                # Fit Model
                grid_search.fit(X_res, y_res)
                # Predict on the kth fold
                print("Testing Random Forest within cross validation - step:" + str(i + 1))
                self.cv_score= grid_search.score(
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, ~self.train_data.columns.isin(self.target_variable)],
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, self.train_data.columns.isin(self.target_variable)])

            if classifier == "svm":
                print("Training SVM within cross validation - step:" + str(i + 1))
                clf = SVC()
                grid_search = GridSearchCV(clf, param_grid= param_grid)
                # Fit Model
                gnb.fit(X_res, y_res)
                # Predict on the kth fold
                print("Testing SVM within cross validation - step:" + str(i + 1))
                self.cv_score = grid_search.score(
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, ~self.train_data.columns.isin(self.target_variable)],
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, self.train_data.columns.isin(self.target_variable)])

            if classifier == "nvb":
                print("Training Naive Bayes within cross validation - step:" + str(i + 1))
                grid_search = GaussianNB()
                grid_search.fit(X_res, y_res)
                # Predict on the kth fold
                print("Testing Naive Bayes within cross validation - step:" + str(i + 1))
                self.cv_score = grid_search.score(
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, ~self.train_data.columns.isin(self.target_variable)],
                    self.train_data.ix[self.train_data.index[cv_index_test]].iloc[:, self.train_data.columns.isin(self.target_variable)])

            accuracy.append(self.cv_score)

        mean_accuracy = np.mean(accuracy).round(decimals=2)
        print("Mean validation accuracy: ", str(mean_accuracy))


        return mean_accuracy, grid_search

    def pandas_to_dict(self, grid_df, index_id):

        current_grid = grid_df.iloc[index_id, ].to_dict()
        for k in grid_df.columns:
            current_grid[k] = [current_grid[k]]

        return current_grid


    def parameter_iterator(self, grid_id, sampling_technique, classifier):

        train_accuracy_vector = []
        predict_accuracy_vector = []
        model_vector = []

        if grid_id is not None:
            # Generate all grid search combinations
            od = OrderedDict(sorted(grid_id.items()))
            cart = list(it.product(*od.values()))
            grid_df = pd.DataFrame(cart, columns=od.keys())
            print("Number of combinations: " + np.str(grid_df.shape[0]))
            k = 0
            for i in range(grid_df.shape[0]):

                print("Grid Search iteration:", str(i+1))
                current_grid = self.pandas_to_dict(grid_df, index_id=i)
                accuracy_results, model_object = self.cv_executor(sampling_technique=sampling_technique,
                                                                  classifier=classifier,
                                                                  param_grid=current_grid
                                                                  )

                train_accuracy_vector.append(accuracy_results)
                model_vector.append(model_object)

                # Perform Model Prediction
                #  -----------------------------------------------------------------------------------------------------------------
                predict_accuracy_vector.append(
                    model_vector[k].score(self.test_data.iloc[:, ~self.test_data.columns.isin(self.target_variable)],
                                          self.test_data[self.target_variable]
                                          ))
                k = k + 1;

            grid_df["Validation Accuracy"] = train_accuracy_vector
            grid_df["Test Accuracy"] = predict_accuracy_vector

        else:
            grid_df = pd.DataFrame()
            accuracy_results, model_object = self.cv_executor(sampling_technique=sampling_technique,
                                                              classifier=classifier,
                                                              param_grid=None
                                                              )
            train_accuracy_vector = accuracy_results
            model_vector = model_object
            predict_accuracy_vector = model_vector.score(self.test_data.iloc[:, ~self.test_data.columns.isin(self.target_variable)],
                                                         self.test_data[self.target_variable]
                                                         )
            grid_df["Validation Accuracy"] = train_accuracy_vector
            grid_df["Test Accuracy"] = predict_accuracy_vector

        return grid_df, model_vector



