"""
Created on Wed Sep 12 22:07:29 2018

@author: salilc
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,confusion_matrix,roc_curve

class ModelGenerator:
    
    # Define the Logistic Regression and Random Forest models.
    def create_tune_models(self,x_train,y_train):
        models = {}
        lr = LogisticRegression(class_weight='balanced',random_state=31)
        rf = RandomForestClassifier(n_estimators=75,max_features=5,max_depth=20,min_samples_split=100,class_weight='balanced',random_state=111)
        xg = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, 
                               learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10,random_state=41)
        lgtbm = lgbm.LGBMClassifier(boosting_type='gbdt', objective='binary',
                       num_iteration=1000,num_leaves=31,
                       is_enable_sparse='true',tree_learner='data',min_data_in_leaf=600,max_depth=4,
                       learning_rate=0.01, max_bin=255, subsample_for_bin=5000, 
                       min_split_gain=5, min_child_weight=5, min_child_samples=10, subsample=0.995, 
                       subsample_freq=1, colsample_bytree=1, reg_alpha=0, 
                       reg_lambda=0, seed=0, nthread=-1, silent=True,random_state=43)
        models["LogisticRegression"] = lr
        models["RandomForest"] = rf
        models["XGBoost"] = xg
        models["LightGBM"] = lgtbm
        for k,v in models.items():
            v.fit(x_train, y_train)
            models[k]=v
        return models
    
    # Train the models on the 10-Fold Cross Validation and calculate AUC.
    def cv_models(self,models,x_train,y_train,kfold,metric):
        model_auc = {}
        for k,v in models.items():
            model_results = model_selection.cross_val_score(v, x_train,y_train, cv=kfold, scoring=metric)
            # print ("Model results: ", model_results)
            mean_auc = model_results.mean()
            std = model_results.std()
            # print out the mean and standard deviation of the training score 
            print('The model {} has AUC {} and STD {}.'.format(k, mean_auc, model_results.std()))
            model_auc[k] = mean_auc
        return model_auc
    
    # Calculate and print model accuracy,confusion matrix and plot ROC curves.
    def display_results(self,model,model_auc,x_test,y_test):          
            print ('\n ---Model Summary---')
            plt.figure()
            for k,v in model.items():
                model_predicted = v.predict(x_test)
                print ('Model accuracy for {} = {}'.format(k,accuracy_score(y_test,model_predicted)))      
                model_roc_auc = roc_auc_score(y_test, model_predicted)
                print ('Model ROC AUC for {} = {}'.format(k,model_roc_auc))
                print(classification_report(y_test, model_predicted))
                model_matrix = confusion_matrix(y_test, model_predicted)
                print('Confusion Matrix for model {} : \n {}'.format(k,model_matrix))
                                                
                fpr, tpr, thresholds = roc_curve(y_test, v.predict_proba(x_test)[:,1])           
                # plot ROC
                plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (k, model_auc[k]))       
            # plot Base Rate ROC
            plt.plot([0,1], [0,1],label='Base Rate')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Plot')
            plt.legend(loc="lower right")
            plt.show()

    # Print feature importance for models.
    def get_feature_importance(self,models,x_train):
        for k,v in models.items():
            if hasattr(v, 'feature_importances_'):
                feature_importances = pd.DataFrame(v.feature_importances_,
                                           index = x_train.columns,
                                            columns=['importance']).sort_values('importance', ascending=False)
                feature_importances = feature_importances.reset_index()
                print "Feature importances for model {} are \n {}".format(k,feature_importances)
                feature_importances.plot.bar()
                plt.show()    
            else:
                print "Feature importances do not exist for model",k