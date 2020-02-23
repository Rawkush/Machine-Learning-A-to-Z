#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:50:09 2020

@author: aarav
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset= pd.read_csv('Salary_Data.csv')



#including all rows and all independent variables coloums
                #row,col
x=dataset.iloc[:,:-1].values
xx=dataset.iloc[:,0].values
# all dependent variable vector
y= dataset.iloc[:,1].values


# spliting dataset to training and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y, test_size=1/3, random_state=0)




'''
no need for feature scaling as lilbrary takes care of it automatically
'''

#Fitting Simpe Linear Regression to Training set
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(x_train,y_train)
# predicting the test set results

y_pred= regressor.predict(x_test) 

# visualising the training set result
plt.scatter(x_train, y_train, color ='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('salary vs Experience(Training set)')
plt.xlabel('year of exp')
plt.ylabel('salary')
plt.show()
         


# visualising the test set result
plt.scatter(x_test, y_test, color ='red')
plt.plot(x_test, y_pred, color='blue')
plt.title('salary vs Experience(Training set)')
plt.xlabel('year of exp')
plt.ylabel('salary')
plt.show()


         


