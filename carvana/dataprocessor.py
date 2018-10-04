
"""
Created on Tue Jul 24 19:05:47 2018

@author: salilc
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
from sklearn import preprocessing

class DataProcessor:
    # Split predictors into categorical and continuous features.
    def get_cat_cont_cols(self,df,cols):
        cat_cols = []
        cont_cols = []
        for col in cols:
            if df[col].dtype == 'object':
                cat_cols.append(col)
            else:
                cont_cols.append(col)
        return cat_cols,cont_cols
    
    # Find null and duplicate values from data frame.
    def findNullValues(self,df):
        null_catcol = []
        null_contcol = []
        null_vals = df.isnull().sum().sort_values()
        df_null = pd.DataFrame({'nullcols':null_vals.index, 'countval':null_vals.values})
        df_null = df_null[df_null.countval > 0]
        print ("Null features with values :",df_null)
        print ("Duplicated values :", df_null.duplicated().sum())
        null_catcol,null_contcol = self.get_cat_cont_cols(df,df_null.nullcols)
        return null_catcol,null_contcol
    
    # Fill NA values with appropirate values for categorical and continuos features
    def fillNAvalues(self,df,null_catcols,null_contcols):
        df_nullcatcols =  df[null_catcols].fillna('NA')
        df_nullcontcols = df[null_contcols]
        df_nullcontcols.fillna(df_nullcontcols.mean(),inplace=True)
        #my_imputer = SimpleImputer()
        #imputed_df_x_cont = my_imputer.fit_transform(df_x[nullcontcols])
        colns = list(set(df.columns) - set(null_catcols) - set(null_contcols))
        df_nafill = pd.concat([df[colns],df_nullcatcols,df_nullcontcols],axis=1)
        return df_nafill


    # Find outliers in continuous features and normalize all featurea into log scale
    def get_outliers_scale(self,df,cols):
        for col in cols:
            stat = df[col].describe()
            # print(stat)
            IQR = stat['75%'] - stat['25%']
            upper = stat['75%'] + 1.5 * IQR
            lower = stat['25%'] - 1.5 * IQR
            print('The upper and lower bounds of {} for suspected outliers are {} and {}.'.format(col,upper, lower))
            print "Values less than lower bound :" , len(df[df[col] < lower])
            print "Values greater than upper  bound : ", len(df[df[col] > upper])
            # converting to log scale
            df[col]=np.log1p(df[col])
        return df[cols]
            
    # Label encode categorical features
    def encode_label(self,df,cols):
        le = preprocessing.LabelEncoder()
        for col in cols:
            df[col] = np.log1p(le.fit_transform(df[col]))
        return df[cols]