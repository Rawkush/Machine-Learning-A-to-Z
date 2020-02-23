#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:28:53 2020

@author: aarav
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoorting dataset
dataset= pd.read_csv('50_Startups.csv')


#including all rows and all independent variables coloums
                #row,col
x=dataset.iloc[:,:-1].values

# all dependent variable vector
y= dataset.iloc[:,4].values

#Encoding chatacyters to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])

#one hot encodeing
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.float64)

#Avoiding dummy variable trap

x=x[:,1:]


# spliting dataset to training and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=0)



#Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(x_train,y_train)
# predicting the test set results

y_pred= regressor.predict(x_test) 

#building optimized the model using 

'''backward elimination'''
import statsmodels.api as sm
'''
y= b0 + b1x1 + b2x2 + ......bnxn
 in the above library b0x0 is not considered i.e forst coloumn is ignored
 so we appedned a coloumn with value 1 in the dataset so this coloumn will
 get ignored not the important one

'''
x= np.append(arr= np.ones((50,1)).astype(int), values=x, axis=1)

'''
Steps of backward elemination

1. select significance level
2. Fit full model with all possible predictors
3. Consider the predictor with highest p value
4. Remove the predictor
5. Fit model without this variable
6. repeat step 3,4,5 until P> significance level

6. Your model is ready

'''

x_opt= x[:,[0,1,2,3,4,5]] # same as x[:, :]

regressor_OLS= sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#removing the highest p value coloum

x_opt= x[:,[0,1,3,4,5]] # same as x[:, :]

regressor_OLS= sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
###################

x_opt= x[:,[0,3,4,5]] # same as x[:, :]

regressor_OLS= sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()
##################3

x_opt= x[:,[0,3,5]] # same as x[:, :]

regressor_OLS= sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

'''
Automatic implementation of above code

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


'''