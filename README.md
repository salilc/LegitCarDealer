# datasciencews

The carvana dataset and the problem statement was obtained from Kaggle through https://www.kaggle.com/c/DontGetKicked/data .

# **Problem Statement**:

One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers. The auto community calls these unfortunate purchases "kicks".

Kicked cars often result when there are tampered odometers, mechanical issues the dealer is not able to address, issues with getting the vehicle title from the seller, or some other unforeseen problem. Kick cars can be very costly to dealers after transportation cost, throw-away repair work, and market losses in reselling the vehicle.

Modelers who can figure out which cars have a higher risk of being kick can provide real value to dealerships trying to provide the best inventory selection possible to their customers.

![Car Odometer](https://psmag.com/.image/t_share/MTI3NTgyNTUyMjA5OTg4MDYy/odometer.jpg)


# **Dataset description**

The challenge of this competition is to predict if the car purchased at the Auction is a good / bad buy (Kick).
The data contains missing values.
The dependent variable (IsBadBuy) is binary.
There are 32 Independent variables.
The data set is split to 60% training and 40% testing.

# **Code walkthrough**
The model is divided into 3 main componenta :

1. The dataprocessor.py deals with data preprocessing viz.[Data Cleaning & Preprocessing](http://nbviewer.jupyter.org/github/salilc/LegitCarDealer/blob/master/carvana.ipynb#Data-Cleaning%20&%20Preprocessing)

      - cleaning data

      - removing unwanted features

      - finding null values and replacing them using appropriate methods.

      - Label encoding categorical features

      - Converting all features to log scale
     

2. The visualizer.py viz.[Data Visualization](http://nbviewer.jupyter.org/github/salilc/datasciencews/blob/master/carvana.ipynb#Data-Visualization) gives us visualizations across predictor variables (both categorical and continuous)

3. The modelgenerator.py file viz.[Model Tuning & Evaluation](http://nbviewer.jupyter.org/github/salilc/datasciencews/blob/master/carvana.ipynb#Model-Tuning-and-Evaluation)
creates,tunes and fits classifier models. It also calculates AUC and plots the confusion matrix and the classification report.
   
# **Future Work** :

1. Work on improving the AUC by fine tuning the hyper parameters of xgboost and LightGBM algorithms.
2. Try and leverage outliers and missing values and see their effect on the model AUC.
