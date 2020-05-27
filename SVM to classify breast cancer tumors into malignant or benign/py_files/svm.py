# @Author: Pipe galera
# @Date:   04-03-2020
# @Email:  pipegalera@gmail.com
# @Last modified by:   Pipe galera
# @Last modified time: 04-03-2020



# -*- coding: utf-8 -*-
"""
Data: sklearn
"""

# Import modules
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
cancer_data = datasets.load_breast_cancer()

# Implementation
X = cancer_data.data
y = cancer_data.target
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.2)

classes = ["malignant", "benign"]

# Suport vector Machine
""" Takes forever to compute if you try different kernels
best_score = 0
for i in ["poly", "rbf", "sigmoid"]:
    clf = svm.SVC(kernel = "poly", gamma = "auto")
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy " + "with kernel " + str(i) + " is " + str(acc))
    # Keep higher accuracy
    if acc > best_score:
        best_score = acc
        best_kernel = i
print("Best cccuracy: " + str(best_score) + " reached with " + str(best_kernel) + " Kernel")
"""

clf_svm = svm.SVC(kernel = "linear", C=2, gamma = "scale")
clf_svm.fit(X_train, y_train)
y_predict = clf_svm.predict(X_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)

# K- Nearest Neighbors
best_score = 0
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_predict)
    # Highest Accuracy
    if acc > best_score:
        best_score = acc
        best_neighbors = i

print("Best acccuracy " + str(best_score) + " reached with " + str(best_neighbors) + " Neighbors")
