# -*- coding: utf-8 -*-
"""
Last Edit: Feb 23 2020
Author: Pipe Galera / Roshan Adusumilli
Project: Machine Learning to Predict Stock Prices
Goal: Predict Stock Prices using a Keras LSTM model

LSTMs are an improved version of recurrent neural networks (RNNs) that remember information over long periods of time

Dataset: Tata Global Beverage’s past stock prices
- “Low” represents the lowest share price for the day.
- “Last” represents the price at which the last transaction for a share went through.
- “Close” represents the price shares ended at for the day.

"""

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential

# Load dataset
os.chdir("C:/Users/fgm.si/github/Medium_projects/Machine Learning to Predict Stock Prices")
data = pd.read_csv("raw_data/NSE-TATAGLOBAL.csv")
data.shape
data.head()

# Inverse the dataset to get the timeline right
data = data.iloc[::-1]
data.tail()

# First calculate the mid prices
high_prices = data.loc[:,"High"].values
low_prices = data.loc[:,"Low"].values
mid_prices = (high_prices+low_prices)/2

# Data visualization
plt.figure(figsize = (18,9))
plt.plot(range(data.shape[0]), mid_prices)
plt.xticks(range(0, data.shape[0], 60),data['Date'].loc[::60],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

# Split the dataset
"""
The training data will be the first 75% data points of the time series and the resting 25% will be test data.We are not going to use the split_train_test of sklearn because is a time series, they are not independent events
"""
train_number = round(mid_prices.shape[0]*0.75)

train_data = mid_prices[:train_number]
test_data = mid_prices[train_number:]

# Fixing the shape to a x times 1D matrix

train_data.shape
train_data = train_data.reshape(-1,1)
train_data.shape
test_data = test_data.reshape(-1,1)
test_data.shape

# Data normalization
"""
You need to make sure that the data behaves in similar value ranges throughout the time frame
"""

scaler = MinMaxScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

# Creating the LSTM model

model = Sequential()
model.add(LSTM(units = 50,))

# Incorporating timesteps into data
data
X_train = []
y_train = []

for i in range(60, 2035):
    X_train = np.append(X_train, data.values[i-60:i, 0])
    y_train = np.append(y_train, data.values[i,0])

X_train
y_train
