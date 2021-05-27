import sys
sys.path.append("/home/mason/Capstone/")

from config import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from sklearn.svm import SVR







#-----------------------Load data from CSV it df

# Load the data into a pandas dataframe
df = pd.read_csv(path + r'/SPY.csv')

#-----------------------Subtract by the number of days to predict by

df = df.head(len(df)-forecast_out)

#Create the independent and dependent variables as lists

days = list()
close_prices = list()

#-----------------------Create date data frame and create close price dataframe

df_days = df.loc[:, 'Date']
df_close_price = df.loc[:, 'Close']

#-----------------------Create independent data set and store in list variable

def create_independent():
    days = list()
    
    for i in range(0, len(df_days), 1):
        #Get value from date row
        a = df_days.iloc[i]
        
        #turn value into ordinal number
        a = datetime.strptime(a, '%Y-%m-%d')
        a = a.toordinal()
        #put ordinal number into days list
        days.append(a)
    
    #make list into array with proper shape
    days = np.array(days)
    days = days.reshape(-1,1)

    return days

days = create_independent()   
    


#-----------------------Create dependent data set and store in list variable

for close_price in df_close_price:
    close_prices.append(float(close_price))



#-----------------------actually make the models


#Create and train an SVR model using a linear kernel
lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days,close_prices)


#Create and train an SVR model using a polynomial kernel
poly_svr = SVR(kernel='poly', C=1000.0, degree=1)
poly_svr.fit(days, close_prices)


#Create and train an SVR model using a RBF kernel
rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
rbf_svr.fit(days, close_prices)




#------------------------Create plot of the 3 models


#Plot the models on a graph to see which has the best fit
plt.figure(figsize=(16,8))
plt.scatter(days, close_prices, color = 'black', label='Original Data')
plt.plot(days, rbf_svr.predict(days), color = 'green', label='RBF Model')
plt.plot(days, poly_svr.predict(days), color = 'orange', label='Polynomial Model')
plt.plot(days, lin_svr.predict(days), color = 'purple', label='Linear Model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()




#Plot the models on a graph to see which has the best fit
plt.figure(figsize=(16,8))
plt.scatter(days, close_prices, color = 'black', label='Original Data')
plt.plot(days, rbf_svr.predict(days), color = 'green', label='RBF Model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()



#actual future predict



day = [[future+200000]]
print('The RBF SVR predicted:', rbf_svr.predict(day))
print('The Linear SVR predicted:', lin_svr.predict(day))
print('The Polynomial SVR predicted:', poly_svr.predict(day))

wye = rbf_svr.predict(day)

print(wye)


