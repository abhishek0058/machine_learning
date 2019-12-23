#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:57:36 2019
@author: Abhishek Sharma
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.printoptions

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode the The Statue column
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder (categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoide the Dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fitting lienar regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# adding an vector of ones in the starting of X
vector_of_ones = np.ones((50, 1)).astype(int)
X = np.append(arr = vector_of_ones, values = X, axis = 1)


# building an optimal model
import statsmodels.api as sm
X_opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

print(regressor_OLS.summary())







