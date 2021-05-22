#Stocks that you want to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA"]

#Path to save file directory
path = r'./Data/'



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
