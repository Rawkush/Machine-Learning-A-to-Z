#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:04:26 2020

@author: aarav
"""

# Importing The Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values




#Encoding chatacyters to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,1])


labelencoder_x_2=LabelEncoder()
x[:,2]= labelencoder_x_2.fit_transform(x[:,2])

#one hot encodeing
from sklearn.compose import ColumnTransformer
#onehotencoder=OneHotEncoder(drop=[0])
#x=onehotencoder.fit_transform(x).toarray()
#putting one hot encoded value in place of country 
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.float64)
x= x[:, 1:]


# spliting dataset to training and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
sc=sc_x.fit(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

#intialising the ANN
classifier = Sequential()

# adding first layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#adding 2nd hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN model to training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)



# predecting the test set Results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

