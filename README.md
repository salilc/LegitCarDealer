# datasciencews
The carvana dataset was obtained from Kaggle through https://www.kaggle.com/c/DontGetKicked/data
The objective of this analysis is to predict if the car purchased at the Auction is a Lemon (bad buy).

- The dataprocessor.py deals with data preprocessing viz.

1.cleaning data

2.removing unwanted features

3.finding null values and replacing by appropriate values

4,Converting all features to log scale

5.Label encoding categorical features

- The visualizer.py gives us visualizations across predictor variables

- The modelgenerator.py file creates,tunes and fits classifier models. It also calculates AUC and plots the
confusion matrix and the classification report.


