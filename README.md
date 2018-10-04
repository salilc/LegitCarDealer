# datasciencews
The carvana dataset was obtained from Kaggle through https://www.kaggle.com/c/DontGetKicked/data .

The objective of this analysis is to predict if the car purchased at the Auction is a Lemon (bad buy).

1. The dataprocessor.py deals with data preprocessing viz.

      - cleaning data

      - removing unwanted features

      - finding null values and replacing them using appropriate methods.

      - Label encoding categorical features

      - Converting all features to log scale
     

2. The visualizer.py gives us visualizations across predictor variables (both categorical and continuous)

3. The modelgenerator.py file creates,tunes and fits classifier models. It also calculates AUC and plots the
   confusion matrix and the classification report.
   
Future Work :

1. Work on improving the AUC by fine tuning the hyper parameters of xgboost and LightGBM algorithms.
2. Try and leverage outliers and missing values and see their effect on the model AUC.
