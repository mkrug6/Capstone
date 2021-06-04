from datetime import date
from datetime import datetime

#Stocks that you want to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA", "FB", "AAPL",
          "AMZN", "GOOGL", "JPM", "JNJ", "V",
          "NVDA", "HD", "PG", "DIS", "BAC", "ADBE",
          "INTC", "VZ", "CSCO", "NFLX", "PFE", "KO",
          "T", "WMT"]

#How far back to gather data?
year = 2021    
month = 3      
day = 1         

#How many days out to predict 
forecast_out = 1


#Path to data directory
path = r'/home/mason/Capstone/Data/'
save_path = 'r/home/mason/Capstone/Figures/'

#Sets start date for stock datetime
today = date.today()
start = date(year, month, day)
start = start.strftime("%Y-%m-%d")

#Sets end date for stock datetime
end = today.strftime("%Y-%m-%d")


future = datetime.toordinal(today) + forecast_out


#List that will contain forecasted values
ticker_dict = {}

#Used to calculate performance metrics
metrics_dict = {}
#Actual close prices dict
acp = {}


deviation_dict = {}
