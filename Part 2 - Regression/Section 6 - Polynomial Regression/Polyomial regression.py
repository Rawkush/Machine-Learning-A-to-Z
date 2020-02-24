#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:42:50 2020

@author: aarav
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoorting dataset
dataset= pd.read_csv('Position_Salaries.csv')

x= dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# fitting Linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#fitting to polynomial regressionto the dataset

from sklearn.preprocessing import PolynomialFeatures

'''
changing the degree will give more accurate results
'''

x_poly = PolynomialFeatures(degree=3)
x_poly= x_poly.fit_transform(x)
poly_reg= LinearRegression()
poly_reg.fit(x_poly, y)

#visualising the Linear result
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('truth or bluff (Linear Regressionn)')
plt.xlabel('position lvel')
plt.ylabel('Salary')
plt.show()


#visualising the Poly result
plt.scatter(x, y, color='red')
plt.plot(x, poly_reg.predict(x_poly ), color='blue')
plt.title('truth or bluff (Polynomial Regressionn)')
plt.xlabel('position lvel')
plt.ylabel('Salary')
plt.show()



















