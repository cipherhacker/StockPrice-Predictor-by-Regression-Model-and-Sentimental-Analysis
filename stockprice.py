


# Random Forest Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('mercury_TCS.csv')
X = dataset.iloc[:, [1,2,3,4]].values
Y = dataset.iloc[:,5].values
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
    #fitting Simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Model Trained! Calling the predict function to estimate the value  the closing price of a stock!")
    




#regressor.score(X_test,Y_test)
def predict(X_test):
    Y_pred = regressor.predict(X_test)
    return float(Y_pred)
