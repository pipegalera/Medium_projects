# @Author: pipegalera
# @Date:   2020-04-24T10:43:20+02:00
# @Last modified by:   pipegalera
# @Last modified time: 2020-04-24T11:51:34+02:00



import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as pyplot
from matplotlib import style

style.use('ggplot')

#
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2019, 12, 31)

data = web.DataReader('TSLA', 'yahoo', start, end)
data.tail()
