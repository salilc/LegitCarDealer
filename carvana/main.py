"""
Created on Wed Sep 12 22:08:14 2018

@author: salilc
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
from dataprocessor import DataProcessor
from visualizer import Visualizer
from modelgenerator import ModelGenerator
from sklearn import model_selection
from sklearn.model_selection import train_test_split


pd.set_option('display.max_row', 10000)
pd.set_option('display.max_columns', 1000)
           
if __name__ == '__main__':
    
    # Reading datset and removing redundant features
    data_file = "/Users/salil/Downloads/training_car.csv"
    df_read = pd.read_csv(data_file)
    df_read = df_read.drop(['AUCGUART','PRIMEUNIT','Nationality','VNZIP1','VNST','BYRNO','WheelTypeID','PurchDate','VehYear'],axis=1)
    
    data = DataProcessor()
    total_columns = df_read.columns
    catcols,contcols =  data.get_cat_cont_cols(df_read,total_columns)    
    print ("Categorical columns: ", catcols)
    
    uid = ['RefId']
    target = ['IsBadBuy']
    contcols = list(set(contcols) - set(uid) - set(target))
    features = catcols + contcols 
    print ("Numerical columns after target and id removal: ", contcols)
    
    df_read.Transmission[df_read.Transmission == 'Manual'] = 'MANUAL'
    df_read.Color[df_read.Color == 'NOT AVAIL'] = 'NA'
    df_read.Color[df_read.Color == 'OTHER'] = 'NA'
    df_read.TopThreeAmericanName[df_read.TopThreeAmericanName == 'OTHER'] = 'NA'
    
    # Finding values with null categorical and continuous features
    null_cat_cols,null_cont_cols = data.findNullValues(df_read)
    df_new = data.fillNAvalues(df_read,null_cat_cols,null_cont_cols)
    
    # Plotting categorical and continuous features
    display = Visualizer()
    display.plot_corr_matrix(df_new)
    display.plot_cat_features(df_new)
    display.plot_cont_features(df_new,contcols)  
    
    # Plotting outliers for conitnuous features and label encoding categorical features
    df_cont = data.get_outliers_scale(df_new,contcols)
    df_cat = data.encode_label(df_new,catcols)
    
    # training the model and running 10-fold cross validation
    df_train = pd.concat([df_cat,df_cont,df_new[target]],axis=1)  
    x_train, x_test, y_train, y_test = train_test_split(df_train[features],df_train[target],test_size=0.2,random_state=7)
    kfold = model_selection.KFold(n_splits=10)
    metric = 'roc_auc'
    
    # compare the model results
    mg = ModelGenerator()
    models = mg.create_tune_models(x_train,y_train)
    model_auc = mg.cv_models(models,x_train,y_train,kfold,metric)
    mg.display_results(models,model_auc,x_test,y_test)
    mg.get_feature_importance(models,x_train)