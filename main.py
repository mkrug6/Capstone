import sys
sys.path.append("/home/mason/Capstone/")
from config import *
from svm import *
#future_array, create_independent, create_dependent #maybe remove
from download import download_csv
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import datetime
#from sklearn.svm import SVR maybe remove
from plot import model_graph, predict_graph
from metrics import generate_metrics, average_deviation
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


for i in range(0, len(ticker), 1):
    tick = ticker[i]
    print("Download CSV files to Data directory")
    download_csv(start, end, path, tick)
    
    print("Loading data from CSV in data frame")
    df = pd.read_csv(path + tick + '.csv')
    
    print('Creating forecast df for ' + tick)
    #Used in metrics.py
    acp[ticker[i]] = df.iat[-1, 4]
    df = df.head(len(df)-forecast_out)
    df_days = df.loc[:, 'Date']
    df_close_price = df.loc[:, 'Close']   
    
    print("Creating independent variable x 'days'...")
    days = create_independent(df_days)
    days = np.array(days)
    days = days.reshape(-1, 1)
    
    print("Creating dependent variable y 'close price'...")
    close_prices = create_dependent(df_close_price)
    
    print("Creating SVR Model using 'x' and 'y'")
    #Creates the model that will train with our data
    rbf_svr = svm_model(days, close_prices, scale=False)
    
    print("Creating predicted future array")
    #Create future array and make it a 2D array
    future_array = make_future_dates(forecast_out, today)
    
    print("Generating fit chart for: %s" % tick) #make i reference the stock
    
    model_graph(rbf_svr, days, close_prices, tick, save=True, show=False)
    
    print("Generating prediction including future days")
    #Creating x value composed of days and future days
    all_days = np.append(days, future_array)
    
    future_close_prices = rbf_svr.predict(future_array)
    all_close_prices = np.append(close_prices, future_close_prices)
    
    predict_graph(rbf_svr, days, close_prices, future_array, future_close_prices, all_days, all_close_prices, tick, save=True, show=False)
    
    
    statement = 'The predicted price is ' + str(future_close_prices) + '.'
    ticker_dict[tick] = statement
    
    metrics_dict[ticker[i]] = future_close_prices[0]
    print()


generate_metrics(acp)

mu = average_deviation()
