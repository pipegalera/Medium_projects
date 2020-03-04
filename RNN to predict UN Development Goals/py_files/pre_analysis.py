# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:40:15 2020

Author: Pipe Galera

Project: Data Driven UN Development Goals

Goal: Predict a specific indicator for each of these goals in 2008 and 2012. Unsupervised Learning

"""

# Import modules
#%reset
import pandas as pd
import zipfile
import sklearn
from sklearn import linear_model,preprocessing
import numpy as np
import os
import pickle

# Load data

os.chdir("C:/Users/fgm.si/Documents/GitHub/data_driven_UN_goals")
_ = zipfile.ZipFile("raw_data/raw_data.zip")
train_data = pd.read_csv(_.open("TrainingSet.csv"), index_col = 0)
submission = pd.read_csv("raw_data/submission.csv", index_col = 0)

# Cleanning column names

_ = dict(zip(train_data.columns[0:-3], range(1972,2008)))
train_data = train_data.rename(_, axis = 1)

# Missing data

train_data.isnull().sum()
train_data.dtypes
train_data.describe()

# Let's keep only the data we are interested

train_data = train_data.loc[submission.index, :]

# Change the order of the columns

_ = train_data.columns.tolist()
_ =  _[-3:] + _[:-3]
train_data = train_data[_]

# Change the first 3 columns to categorical 

le = LabelEncoder()

train_data["Country Name"] = le.fit_transform(train_data["Country Name"])
train_data["Series Code"] = le.fit_transform(train_data["Series Code"])
train_data["Series Name"] = le.fit_transform(train_data["Series Name"])

# Imputing using interpolation with method: linear

train_data.isnull().sum()

train_data.interpolate(
                    method = "linear", 
                    inplace = True,
                    limit_direction = "both",
                    axis = 0
                    )
train_data.isnull().sum()

# Linear Model

predict = 2007
X = np.array(train_data.drop(predict, axis = 1))
y = np.array(train_data[predict])

# Train the model and keep the best score

best = 0
for _ in range(100):
    #Running a 100 regressions
    
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X_train, y_train)
    acc = linear_regression.score(X_test, y_test)
    print("Accuracy:" + str(acc))
    
    #Keeping the best score
    if acc > best:
        best = acc
        # Saving Our Model
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear_regression, f)

# Viewing The Constants
            
print("-------------------------")
print( "Coefficients: \n", linear_regression.coef_)
print("-------------------------")
print("Intercept \n", linear_regression.intercept_)
print("-------------------------")



