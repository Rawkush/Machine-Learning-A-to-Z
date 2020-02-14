# -*- coding: utf-8 -*-
"""
Spyder Editor
press ctrl + I after selecting a class to get info about it
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoorting dataset
dataset= pd.read_csv('Data.csv')

#including all rows and all independent variables coloums
                #row,col
x=dataset.iloc[:,:-1].values

# all dependent variable vector
y= dataset.iloc[:,-1].values


# handling missing data by putting mean value of that coloum
from sklearn.impute import SimpleImputer 
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])


#Encoding chatacyters to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

#one hot encodeing
from sklearn.compose import ColumnTransformer
#onehotencoder=OneHotEncoder(drop=[0])
#x=onehotencoder.fit_transform(x).toarray()

    #putting one hot encoded value in place of country 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.str)

#Encoding dependent varible coloum to numbers

labelencoder_y =LabelEncoder()
y=labelencoder_y.fit_transform(y)


# spliting dataset to training and test set

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
sc=sc_x.fit(x_test)

x_train= sc.transform(x_train)
x_test= sc.transform(x_test)




















