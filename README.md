# datasciencews
The carvana dataset was obtained from Kaggle through https://www.kaggle.com/c/DontGetKicked/data
The objective of this analysis is to predict if the car purchased at the Auction is a Lemon (bad buy).

1. The dataprocessor.py deals with data preprocessing viz.

- cleaning data

- removing unwanted features

- finding null values and replacing by appropriate values

- Converting all features to log scale

- Label encoding categorical features


2. The visualizer.py gives us visualizations across predictor variables

3. The modelgenerator.py file creates,tunes and fits classifier models. It also calculates AUC and plots the
confusion matrix and the classification report.


