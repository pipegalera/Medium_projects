# @Author: pipegalera
# @Date:   2020-03-12T17:45:30+01:00
# @Last modified by:   pipegalera
# @Last modified time: 2020-03-12T17:48:52+01:00



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
os.chdir("/Users/pipegalera/Documents/GitHub/side_projects/RNN to predict UN Development Goals")
data = pd.read_csv("out_data/train_data_with_missing.csv")
data.shape
data.head(2)
