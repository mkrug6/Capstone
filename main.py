import sys
sys.path.append("/home/mason/Capstone/")
from config import *
from svm import * #maybe remove
from download import download_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from plotter import model_graph, predict_graph

print("Download CSV files to Data directory")
download_csv(start, end, ticker, path)







#Load data from CSV and put in df
df = pd.read_csv(path + r'/SPY.csv')
   
#Subtract by the number of days to predict by
df = df.head(len(df)-forecast_out)

#Create x and y dataframes
df_days = df.loc[:, 'Date']
df_close_price = df.loc[:, 'Close']

#-----------------------Create date data frame and create close price dataframe


days = create_independent()
days = np.array(days)
days = days.reshape(-1, 1)


close_prices = create_dependent()


#Creates the model that will train with our data
rbf_svr = svm_model(days, close_prices, scale=False)


print(rbf_svr)


#Create future array and make it a 2D array
future_array = np.array(f_list)
future_array = future_array.reshape(-1, 1)


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
