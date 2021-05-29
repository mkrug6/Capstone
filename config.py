from datetime import date
from datetime import datetime

#Stocks that you want to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA", "FB", "AAPL"]

#How far back to gather data?
year = 2021     #Dummy  value for testing
month = 3       #Dummy  value for testing
day = 1         #Dummy  value for testing

#How many days out to predict 
forecast_out = 1


#Path to data directory
path = r'/home/mason/Capstone/Data/'


#Sets start date for stock datetime
today = date.today()
start = date(year, month, day)
start = start.strftime("%Y-%m-%d")

#Sets end date for stock datetime
end = today.strftime("%Y-%m-%d")


future = datetime.toordinal(today) + forecast_out


#List that will contain forecasted values
ticker_dict = {}
