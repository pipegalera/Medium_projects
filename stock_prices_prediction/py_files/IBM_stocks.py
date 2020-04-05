GRU_rmse# @Author: pipegalera
# @Date:   2020-03-12T15:30:58+01:00
# @Last modified by:   Pipe galera
# @Last modified time: 05-04-2020



# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
plt.style.use('seaborn-deep')
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
from math import sqrt

# Load dataset
os.chdir("/Users/fgm.si/Documents/GitHub/side_projects/stock_prices_prediction")
data = pd.read_csv('raw_data/IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
data.shape
data.head(2)

# Split of train and test
data[:"2016"].iloc[:,1:2].values.shape
data["2017":].iloc[:,1:2].values.shape

train = data[:"2016"].iloc[:,1:2].values
test = data["2017":].iloc[:,1:2].values

# Visualization
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(data["High"][:"2016"], label="Training set - 2006 to 2017")
ax.plot(data["High"]["2017":], label= "Test set - After 2017")
ax.legend(loc="upper right")
ax.set(title = "IBM stock price, from 2006 to 2018",
       xlabel = "Date",
       ylabel = "High stock price of the day")
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
plt.savefig("figures/ibm_stock_price")

# Train data normalization with MinMaxScaler
mms = MinMaxScaler()
train_mms = mms.fit_transform(train)

# timesteps for the Train data
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

# np.reshape(a, newshape, order='C')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_train.shape

#########################################
# Long Short Term Networks Model (LSTM) #
#########################################

# Creating the LSTM model

"""
The LSTM layer is added with the following  arguments: 50 units is the dimensionality of the output space, "return_sequences=True" is necessary    forstacking LSTM layers so the consequent LSTM layer has a three-dimensional sequence input, and "input_shape" is the shape of the training dataset.

Specifying 0.2 in the Dropout layer means that 20% of the layers will be dropped, to avoid vanishin or exploding gradients.

"""
LSTM_model = Sequential()
# First LSTM layer with Dropout regularisation
LSTM_model.add(LSTM(
    units = 50,
    return_sequences = True,
    input_shape = (X_train.shape[1], 1),
))
LSTM_model.add(Dropout(0.2))
# Second LSTM layer
LSTM_model.add(LSTM(units=50,return_sequences=True))
LSTM_model.add(Dropout(0.2))
# Third LSTM layer
LSTM_model.add(LSTM(units=50,return_sequences=True))
LSTM_model.add(Dropout(0.2))
# Fourth LSTM layer
LSTM_model.add(LSTM(units=50))
LSTM_model.add(Dropout(0.2))
# The output layer
LSTM_model.add(Dense(units=1))

LSTM_model.compile(optimizer='rmsprop',loss='mean_squared_error')

"""
we fit the LSTM_model to run for 50 epochs, the number of times the learning algorithm will work through the entire training set, with a batch size of 32.
"""
LSTM_model.fit(X_train, y_train, epochs = 50, batch_size = 32)

# We first calculate how many inputs we need
n_inputs = len(data["High"]) - len(test) - 60
n_inputs

# Calculate the inputs
inputs = data["High"][n_inputs:].values
inputs = inputs.reshape(-1,1)
inputs.shape

# Test data normalization with MinMaxScaler
inputs_mms = mms.transform(inputs)
inputs_mms.shape

# Timesteps for the Test data
X_test = []
for i in range(60,inputs_mms.shape[0]):
    X_test.append(inputs_mms[i-60:i, 0])

# In order to reshape, first they have to be in arrays
X_test = np.array(X_test)

# Reshape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Prediction
LSTM_pred_stock_price = LSTM_model.predict(X_test)
LSTM_pred_stock_price = mms.inverse_transform(LSTM_pred_stock_price)

# Visualization of the results

# First, we index test and the prediction values with the last days of the dataset.
pred_days = LSTM_pred_stock_price.shape[0]
index_days = data.index[-pred_days:]

LSTM_pred_stock_price_df  = pd.DataFrame(LSTM_pred_stock_price)
LSTM_pred_stock_price_df = LSTM_pred_stock_price_df.set_index(index_days)

test_df = pd.DataFrame(test)
test_df = test_df.set_index(data.index[-pred_days:])
test_df

# Now we plot the results with the correct date axis
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(test_df, color='red', label='Real IBM Stock Price')
ax.plot(LSTM_pred_stock_price_df, color='blue',label='Predicted IBM Stock Price')
ax.legend(loc="upper right")
ax.set(title = 'IBM Stock Price Prediction using LSTM',
       xlabel = "Date",
       ylabel = "High stock price of the day")
plt.savefig("figures/LSTM_ibm_stock_price_pred.jpg")


plt.plot(test, color='red',label='Real IBM Stock Price')
plt.plot(LSTM_pred_stock_price, color='blue',label='Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
#plt.show()
plt.savefig("figures/LSTM_ibm_stock_price_pred.jpg")

# Calculating accuracy
LSTM_rmse = sqrt(mean_squared_error(test, LSTM_pred_stock_price))
round(LSTM_rmse, 2)


###############################
# Gated Recurrent Units (GRU) #
###############################

# Creating the GRU model

GRU_model = Sequential()
# First GRU layer with Dropout regularisation
GRU_model.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Second GRU layer
GRU_model.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Third GRU layer
GRU_model.add(GRU(units=50,
                    return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Fourth GRU layer
GRU_model.add(GRU(units=50,
                    activation='tanh'))
GRU_model.add(Dropout(0.2))
# The output layer
GRU_model.add(Dense(units=1))
# Compiling the RNN
GRU_model.compile(optimizer=SGD(lr=0.01,
                                    decay=1e-7,
                                    momentum=0.9,
                                    nesterov=False),
                     loss='mean_squared_error')
# Fitting to the training set
GRU_model.fit(X_train,y_train,epochs=50, batch_size=32)

# Predict prices
pred_stock_price_GRU = GRU_model.predict(X_test)
pred_stock_price_GRU = mms.inverse_transform(pred_stock_price_GRU)
pred_stock_price_GRU

# Visualization of the results

# First, we index test and the prediction values with the last days of the dataset.
pred_stock_price_GRU_df  = pd.DataFrame(pred_stock_price_GRU)
pred_stock_price_GRU_df = pred_stock_price_GRU_df.set_index(index_days)

# Now we plot the results with the correct date axis
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(test_df, color='red', label='Real IBM Stock Price')
ax.plot(pred_stock_price_GRU_df, color='green',label='Predicted IBM Stock Price')
ax.legend(loc="upper right")
ax.set(title = 'IBM Stock Price Prediction using GRU',
       xlabel = "Date",
       ylabel = "High stock price of the day")
plt.savefig("figures/GRU_ibm_stock_price_pred.jpg")


# Calculating accuracy
GRU_rmse = sqrt(mean_squared_error(test, pred_stock_price_GRU))
round(GRU_rmse, 2)

# Visualization of both results

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(test_df, color='red', label='Real IBM Stock Price')
ax.plot(pred_stock_price_GRU_df, color='green',label='Predicted IBM Stock Price using GRU. RMSE = 3.23')
ax.plot(LSTM_pred_stock_price_df, color='blue',label='Predicted IBM Stock Price using LSTM. RMSE = 3.01')
ax.legend(loc="upper right")
ax.set(title = 'IBM Stock Price Prediction',
       xlabel = "Date",
       ylabel = "High stock price of the day")
plt.savefig("figures/stock_price_pred.jpg")
