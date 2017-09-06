import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import cPickle as pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class prepare_MD2K_data(BaseEstimator, TransformerMixin):
    '''Prepares features from CSV file and 

    Parameters
    ----------
    numeric_features_fields: path to txt file or (TODO) list of numeric feature file names
    cat_features_fields: path to txt file or (TODO) list of categorical feature file names
    
    Attributes
    ----------
    numeric_feature_field_names: list of numeric column names
    cat_feature_field_names: list of categorical column names
    
    
    Examples
    --------
    '''    
    def __init__(self, numeric_features_fields, cat_features_fields):
        with open(numeric_features_fields) as file:
            self.numeric_feature_field_names = [line.strip() for line in file]
        with open(cat_features_fields) as file:
            self.cat_feature_field_names = [line.strip() for line in file]
            
    def _get_time_since(self, activity_indicator):
        time_since = [activity_indicator[0]]
        for activity in activity_indicator[1:]:
            if activity == 1:
                time_since.append(1)
            elif time_since[-1] == 0:
                time_since.append(0)
            else:
                time_since.append(time_since[-1]+1)
        return np.array(time_since)            
            
    def fit(self, X = None, y = None):        
        # load data
        df = pd.read_csv(X)
        # sort by time index 
        df.dateTime = pd.to_datetime(df.dateTime)
        
        # get relevant subset
        df_rel = df[self.numeric_feature_field_names+ self.cat_feature_field_names]
        
                
        # deal with categorical features
        dfs_list = []
        for cat_feature in self.cat_feature_field_names:
            dfs_list.append(pd.get_dummies(df_rel[cat_feature]))
            df_rel.drop(cat_feature, 1, inplace = True)
        dfs_list.append(df_rel)    
        df_rel = pd.concat(dfs_list, axis = 1)            
        
        # compute time since activities
        activities = pd.unique(df["prediction"])

        for activity in activities:
            #df_rel[["time.since." + str(activity)]] = np.array(self._get_time_since(df_rel[activity].tolist()))
            df_rel.loc[:,"time.since." + str(activity)] = np.array(self._get_time_since(df_rel[activity].tolist()))
            #df_rel = df_rel.assign("time.since." + str(activity) = self._get_time_since(df_rel[activity].tolist())
            
        self.df_features = df_rel
       
        #return self
    
    def transform(self, X = None):
        return self.df_features

def get_auc(train_paths, test_path, 
            numeric_features_path = "numeric_features_fields.txt",
            cat_features_path = "categorical_features_fields.txt",
            classifier = LogisticRegressionCV(class_weight= "balanced", n_jobs=-2, scoring = 'roc_auc')):
    
    #---------------------------------#
    # instantiate MD2K data prep object
    #---------------------------------#
    prep_data = prepare_MD2K_data(numeric_features_fields = numeric_features_path, 
                              cat_features_fields = cat_features_path)
    
    #---------------------------------#
    # prep test set
    #---------------------------------#  
    # set the target variable from test path and only read 'Eat' column
    y_test = np.array(pd.read_csv(test_path,usecols=['Eat']))
    
    # "fit" and feature engineer the data (eg create time since * variables)
    # impute missing values with means
    prep_data.fit(test_path)
    X_test_df = prep_data.transform(test_path)
    X_test_df.fillna(X_test_df.mean(), axis = 'index', inplace=True)
    X_test = np.array(X_test_df*1.)
    
    #-------------------------------------------#
    # prep train set using MD2K data prep object
    # that was initialized using the test set
    #-------------------------------------------#    
    X_train_list = []
    y_train_list = []
    for train_path in train_paths:
        prep_data.fit(train_path)
        X_train_list.append(prep_data.transform(train_path))
        y_train_list.append(np.array(pd.read_csv(train_path, usecols=["Eat"])))
        
    # stack up train labels and feautres into one big array
    y_train = np.vstack(y_train_list)
    X_train_df = pd.concat(X_train_list, axis = 0)[X_test_df.columns]
    X_train_df.fillna(X_train_df.mean(), axis = 'index', inplace=True)
    X_train = np.array(X_train_df*1.)
    
    #-------------------------------------------#
    # train and evaluate with Logistic classifier
    #-------------------------------------------#     
    #clf = LogisticRegressionCV(Cs = np.logspace(-4,9, 3), class_weight= "balanced",  n_jobs=-2, scoring = 'roc_auc')
    #clf = LogisticRegressionCV(**kwargs)
    #clf.set_params(**kwargs)
    classifier.fit(X_train, y_train.squeeze())
    
    return roc_auc_score(y_test.squeeze(),classifier.predict_proba(X_test)[:,1])
