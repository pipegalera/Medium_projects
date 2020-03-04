# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:40:15 2020
Author: Pipe Galera
Project: Introduction to KNN
Goal: classify cars in 4 categories based upon certain features.
Data: https://archive.ics.uci.edu/ml/datasets/car+evaluation
"""

%reset

# Import modules

import pandas as pd
import numpy as np
import zipfile
import os
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier

# Read the data

os.chdir("C:/Users/fgm.si/github/ML-AI-Megacourse/2. K-Nearest Neighbors")
data = pd.read_csv("raw_data/car.data")

# Pre-Analysis and cleanning irregular data

data.isnull().sum()
data.dtypes

le = preprocessing.LabelEncoder()

buying = le.fit_transform(data["buying"].values.reshape(-1,1))
maint = le.fit_transform(data["maint"].values.reshape(-1,1))
door = le.fit_transform(data["door"].values.reshape(-1,1))
lug_boot = le.fit_transform(data["lug_boot"].values.reshape(-1,1))
safety = le.fit_transform(data["safety"].values.reshape(-1,1))
clas = le.fit_transform(data["class"].values.reshape(-1,1))

# Implementation

X = list(zip(buying, maint, door, lug_boot, safety))
y = list(clas)

best_score = 0
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print("Accuracy " + "with " + str(i) + " neighbors is " + str(acc))

    # Highest Accuracy
    if acc > best_score:
        best_score = acc
        best_neighbors = i

print("Best cccuracy " + str(best_score) + " reached with " + str(best_neighbors) + " Neighbors")

# Seeing actual predictions

predictions = knn.predict(X_test)

cars_classification = ["unacc,", "acc,", "good,", "vgood,"]
for i in range(len(predictions)):
    print("Predicted: ", cars_classification[predictions[i]], "Actual: ", cars_classification[y_test[i]])
