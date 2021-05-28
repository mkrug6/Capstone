from sklearn.svm import SVR
from datetime import datetime
import numpy as np
# from config import *
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# import math



def svm_model(days, close_prices, scale=bool):

    # Create and train an SVR model using a RBF kernel
    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
    if scale: 
        print("insert scaling code here")
    rbf_svr.fit(days, close_prices)
    
    return rbf_svr




    # actual future predict

    day = [[future+200000]]
    print('The RBF SVR predicted:', rbf_svr.predict(day))
    print('The Linear SVR predicted:', lin_svr.predict(day))
    print('The Polynomial SVR predicted:', poly_svr.predict(day))




def create_independent(df_days):
    # Create the independent and dependent variables as lists
    days = list()
    for i in range(0, len(df_days), 1):
        # Get value from date row
        a = df_days.iloc[i]

        # turn value into ordinal number
        a = datetime.strptime(a, '%Y-%m-%d')
        a = a.toordinal()
        # put ordinal number into days list
        days.append(a)

    return days


def create_dependent(df_close_price):
   # Create dependent data set and store in list variable
   close_prices = list()
   for close_price in df_close_price:
       close_prices.append(float(close_price))
   return close_prices




def future_array(forecast_out, today):
    f_list = list()
    t = datetime.toordinal(today)
    for i in range(t, t + forecast_out, 1):
        t += 1
        f_list.append(t)
    future_array = np.array(f_list)
    future_array = future_array.reshape(-1, 1)
    return future_array