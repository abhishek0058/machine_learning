#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:14:20 2019

@author: m
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# making linear regression modal
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)

pred_y = lin_reg.predict(X)

# making the polynomial regression
# first we add the polynomial terms and then fit_transform those in
# a Linear_regressor

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

pred_y_poly = lin_reg_2.predict(X_poly)

# making the graph for both regressors, to compare the performance
# Linear regression
plt.scatter(X, y, color = 'red')
plt.plot(X, pred_y, color = 'blue')
plt.title('Truth or Bluf (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Polynomial regression

# high resolution plot

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

pred_y_poly_high_resolution = lin_reg_2.predict(poly_reg.fit_transform(X_grid))

plt.scatter(X, y, color = 'red')

# plt.plot(X, pred_y_poly, color = 'blue')
plt.plot(X_grid, pred_y_poly_high_resolution, color = 'green')

plt.title('Truth or Bluf (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()



# prediction based on linear-regression and polynomial-regression
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

