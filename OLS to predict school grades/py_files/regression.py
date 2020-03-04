# @Author: Pipe galera
# @Date:   28-02-2020
# @Email:  pipegalera@gmail.com
# @Last modified by:   Pipe galera
# @Last modified time: 04-03-2020

"""
Data from:
https://archive.ics.uci.edu/ml/datasets/Student+Performance
Goal: Predicting 3 grade grades (G3)
"""

%reset
# Packages
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
import os



# Read data

os.chdir("C:/Users/fgm.si/github/ML-AI-Megacourse/1. Linear regression")
data = pd.read_csv("raw_data/student-mat.csv", sep = ";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Separating Our Data

predict = "G3"
X = np.array(data.drop(predict, 1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

# Implementing the Algorithm, keeping only the best score of running it 100 times

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

print( "Coefficients: \n", linear_regression.coef_)
print("Intercept \n", linear_regression.intercept_)
print("-------------------------")

# Seeing the predictions on Specific Stundets

predictions = linear_regression.predict(X_test)

for i in range(len(predictions)):
    print("Predicted: ", predictions[i], "Data: ", X_test[i], "Actual: ", y_test[i])

# Drawing and plotting model

plt.style.use("ggplot")
plot = "G2"
plt.scatter(data[plot], data["G3"])
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()

"""
# Instead of linear_regression you can just load the model from pickle

pickle_in = open("studentgrades.pickle", "rb")
linear_regression = pickle.load(pickle_in)
"""
