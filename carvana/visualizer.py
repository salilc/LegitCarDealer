"""
Created on Wed Sep 12 22:07:21 2018

@author: salilc
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:

    # Correlation matrix for continuous features
    def plot_corr_matrix(self,df):
        corr = df.corr()
        # plot the heatmap
        sns.heatmap(corr, 
                xticklabels=corr.columns,
                yticklabels=corr.columns)
        plt.figure(figsize=(2,2))
        plt.show()

    def plot_cat_features(self,df):
        plt.figure(figsize = (14, 6))
        sns.scatterplot(x='MMRAcquisitonRetailCleanPrice',y = 'MMRCurrentAuctionCleanPrice',data=df)
        plt.show()

        plt.figure(figsize = (30, 15))
        sns.lineplot(x='Make',y = 'VehicleAge',data=df)
        plt.show()
        
    def plot_cont_features(self,df,contcols):
        for i in range (1,len(contcols)):
            plt.figure(figsize = (14, 6))
            plt.subplot(1, 2, 1)
            sns.distplot(df[contcols[i]])
            plt.subplot(1, 2, 2)
            sns.boxplot(df[contcols[i]])
            plt.show() 

        plt.figure(figsize = (14, 6))
        sns.stripplot(x='Size',y = 'WarrantyCost',data=df)
        plt.show()

        plt.figure(figsize = (14, 6))
        sns.violinplot(x='Color',y = 'WarrantyCost',data=df)
        plt.show()

        plt.figure()
        sns.countplot(x = "TopThreeAmericanName",data=df)
        plt.show()

        plt.figure()
        sns.countplot(x = "WheelType",data=df)
        plt.show()
        
        plt.figure(figsize = (14, 6))
        sns.boxplot(x='Color',y = 'WarrantyCost',data=df)
        plt.show()

        plt.figure(figsize = (14, 6))
        sns.pointplot(x='Auction',y = 'WarrantyCost',data=df)
        plt.show()

        plt.figure(figsize = (14, 6))
        sns.barplot(x='Color',y = 'VehBCost',data=df)
        plt.show()

        plt.figure(figsize = (14, 6))
        sns.lmplot(x='VehOdo',y = 'WarrantyCost',hue='IsBadBuy',data=df)
        plt.show()