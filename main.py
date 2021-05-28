import sys
sys.path.append("/home/mason/Capstone/")
from config import *
from svm import *
#future_array, create_independent, create_dependent #maybe remove
from download import download_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#from sklearn.svm import SVR maybe remove
from plotter import model_graph, predict_graph
from datetime import datetime, today


print("Download CSV files to Data directory")
download_csv(start, end, ticker, path)







print("#Loading data from CSV in data frame")
df = pd.read_csv(path + r'/SPY.csv')
df_days = df.loc[:, 'Date']
df_close_price = df.loc[:, 'Close']   

###Maybe get rid of
##Subtract by the number of days to predict by
#df = df.head(len(df)-forecast_out)

print("Creating independent variable x 'days'...")
days = create_independent(df_days)
days = np.array(days)
days = days.reshape(-1, 1)

print("Creating dependent variable y 'close price'...")
close_prices = create_dependent(df_close_price)


print("Creating SVR Model using 'x' and 'y'")
#Creates the model that will train with our data
rbf_svr = svm_model(days, close_prices, scale=False)


print(rbf_svr)


print("Creating predicted future array")
#Create future array and make it a 2D array
future_array = future_array(forecast_out, today)

days =9

future_array = np.array(f_list)
future_array = svm.future_array.reshape(-1, 1)


#Predict using just the future array
future_prediction = rbf_svr.predict(future_array)


model_graph(future_array, future_prediction)






#-------------------------Graphs the effectiveness of the model


    future_array = np.array(f_list)
    future_array = future_array.reshape(-1, 1)



model_graph(days, close_prices, save=False)



                    
                    
                    
                    
                    




print(rbf_svr.predict(future_array))













future):
    
    
    
    
    
    
    return something
