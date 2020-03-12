# @Author: Pipe galera
# @Date:   04-03-2020
# @Email:  pipegalera@gmail.com
# @Last modified by:   pipegalera
# @Last modified time: 2020-03-12T17:48:57+01:00



# -*- coding: utf-8 -*-
"""
Author: Pipe Galera
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
plt.style.use('fivethirtyeight')
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
from math import sqrt

# Load dataset
os.chdir("/Users/pipegalera/Documents/GitHub/side_projects/LSTM to predict Stock Prices")
data = pd.read_csv("raw_data/NSE-TATAGLOBAL.csv")
data.shape
data.head(2)

# Inverse the dataset to get the timeline right
data = data.iloc[::-1]
data.head(2)

# Put the date to the right format
data.columns

data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace = True)
data.head(2)
data.shape
"""
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
"""
# Split of train and test
data[:"2017"].iloc[:,1:2].values.shape
data["2018":].iloc[:,1:2].values.shape

train = data[:"2017"].iloc[:,1:2].values
test = data["2018":].iloc[:,1:2].values

# Visualization
data["High"][:"2017"].plot(figsize = (18,9), legend = True)
data["High"]["2018":].plot(figsize = (18,9), legend = True)
plt.legend(["Training set - Before 2018", "Test set - After 2018"])
plt.ylabel('High Price',fontsize=18)
plt.title("TATAS stock price")
plt.show()

# Data normalization with MinMaxScaler
mms = MinMaxScaler()
train_mms = mms.fit_transform(train)

# timesteps for the train set
train_mms.shape

X_train = []
y_train = []

for i in range(60, train_mms.shape[0]):
    # 60 timesteps and 2035 observations
    X_train.append(train_mms[i-60:i, 0])
    y_train.append(train_mms[i,0])

# In order to reshape, first they have to be in arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
y_train.shape

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


# Creating the LSTM model
"""
The LSTM layer is added with the following  arguments: 50 units is thed imensionality of the output space, "return_sequences=True" is necessary    forstacking LSTM layers so the consequent LSTM layer has a three-dimensional sequence input, and "input_shape" is the shape of the training dataset.

Specifying 0.2 in the Dropout layer means that 20% of the layers will be dropped

"""
model = Sequential()
# First LSTM layer with Dropout regularisation
model.add(LSTM(
    units = 50,
    return_sequences = True,
    input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
# Third LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
# Fourth LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# The output layer
model.add(Dense(units=1))
# Compiling the RNN
model.compile(optimizer='adam',loss='mean_squared_error')

"""
we fit the model to run for 100 epochs, the number of times the learning algorithm will work through the entire training set, with a batch size of 32.
"""
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Calculate the number of inputs that we need
n_inputs = len(data["High"]) - len(test) - 60
n_inputs

# Calculate inputs
inputs = data["High"][n_inputs:].values
inputs = inputs.reshape(-1,1)

# Test data normalization with MinMaxScaler
inputs_mms = mms.transform(inputs)
inputs_mms.shape

# Timesteps for the Test data
X_test = []
for i in range(60,inputs_mms.shape[0]):
    X_test.append(inputs_mms[i-60:i, 0])
# In order to reshape, first they have to be in arrays
X_test = np.array(X_test)
X_test.shape
# Reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape

# Predict prices
pred_stock_price = model.predict(X_test)
pred_stock_price = mms.inverse_transform(pred_stock_price)
pred_stock_price


# Visualization of the results
plt.plot(test, color = 'red', label = 'Real TATA Stock Price')
plt.plot(pred_stock_price, color = 'blue', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()

# Calculating accuracy
rmse = sqrt(mean_squared_error(test, pred_stock_price))
rmse

###############################
# Gated Recurrent Units (GRU) #
###############################

# Creating the GRU model

regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50,
                    activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                     loss='mean_squared_error')
# Fitting to the training set
regressorGRU.fit(X_train,y_train,epochs=3, batch_size=150)

# Predict prices
pred_stock_price_GRU = regressorGRU.predict(X_test)
pred_stock_price_GRU = mms.inverse_transform(pred_stock_price_GRU)
pred_stock_price_GRU

# Visualization of the results
plt.plot(test, color = 'red', label = 'Real TATA Stock Price')
plt.plot(pred_stock_price, color = 'blue', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()

# Calculating accuracy
rmse_2 = sqrt(mean_squared_error(test, pred_stock_price_GRU))
rmse_2
