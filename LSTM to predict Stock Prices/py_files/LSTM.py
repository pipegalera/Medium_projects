# @Author: Pipe galera
# @Date:   04-03-2020
# @Email:  pipegalera@gmail.com
# @Last modified by:   Pipe galera
# @Last modified time: 04-03-2020



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
os.chdir("C:/Users/fgm.si/documents/GitHub/side_projects/Machine Learning to Predict Stock Prices")
data = pd.read_csv("raw_data/NSE-TATAGLOBAL.csv")
data.shape
data.head(2)

# Inverse the dataset to get the timeline right
data.head()
data = data.iloc[::-1]
data.tail()

# Put the date to the right format
data.columns

data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace = True)
data.head()
data.shape
# First calculate the "mid prices", or the average between the High and the low stock value
high_prices = data.loc[:,"High"].values
low_prices = data.loc[:,"Low"].values
mid_prices = (high_prices+low_prices)/2


# Data visualization
plt.figure(figsize = (18,9))
plt.plot(range(data.shape[0]), mid_prices)
#plt.xticks(range(0, data.shape[0], 60),data['Date'].loc[::60],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

# Data normalization
"""
You need to make sure that the data behaves in similar value ranges throughout the time frame
"""
# Fixing the shape to a x times 1D matrix
# We are only interested in the mid_prices
mid_prices.shape
data.shape
data_array = mid_prices.reshape(mid_prices.shape[0], 1)

# Data normalization
scaler = MinMaxScaler()
data_array_scalated = scaler.fit_transform(data_array)
data_array_scalated

# Split the dataset
"""
The training data will be the first 75% data points of the time series and the resting 25% will be test data.We are not going to use the split_train_test of sklearn because is a time series, they are not independent events.
"""


mid_prices.shape[0] # Numer of observations
train_number = round(mid_prices.shape[0]*0.75)

train_data = mid_prices[:train_number]
test_data = mid_prices[train_number:]

"""
# Incorporating timesteps into data
X_train = []
y_train = []

for i in range(60, 2035):
    # 60 timesteps and 2035 observations
    X_train = np.append(X_train, train_data[i-60:i, 0])
    y_train = np.append(y_train, train_data[i,0])

X_train.shape

# In order to reshape, first they have to be in arrays
X_train.shape
y_train.shape
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
y_train.shape

# np.reshape(a, newshape, order='C')[source]
X_train = np.reshape(X_train, (X_train.shape[0], 1) ,1)
"""

# Creating the LSTM model
    """
    The LSTM layer is added with the following  arguments: 50 units is the dimensionality of the    output space, return_sequences=True is necessary    for stacking LSTM layers so the consequent LSTM     layer has a three-dimensional sequence input, and   input_shape is the shape of the training dataset.

    Specifying 0.2 in the Dropout layer means that 20% of the layers will be dropped

    """
model = Sequential()
model.add(LSTM(
    units = 50,
    return_sequences = True,
    input_shape = (X_train.shape[1], 1),
))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

"""
we fit the model to run for 100 epochs, the number of times the learning algorithm will work through the entire training set, with a batch size of 32.
"""
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train, epochs = 100, batch_size = 32)


#####################################################

"""
We want to predict the n days ahead (foward_days) having as input the m past observed days (look_back). So, if we have an input of m past days, the network output will be the prediction for the n next days.

The test data will be composed of k periods (num_periods), in which every period is a series of n days prediction
"""
forward_days = 10
look_back = 40
num_periods = 20

num_neurons_FL = 128
num_neurons_SL = 64
epochs = 220

model = Sequential()
model.add(LSTM(
    units = num_neurons_FL,
    return_sequences = True,
    input_shape = (look_back, 1),
))
model.add(Dropout(0.2))

model.add(LSTM(
    units = num_neurons_SL,
    return_sequences = True,
    input_shape = (num_neurons_FL, 1),
))
model.add(Dropout(0.2))
model.add(Dense(forward_days))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train, epochs = epochs,
                            shuffle = True,
                            batch_size = 2,
                            verbose = 2)
