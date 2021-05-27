from datetime import date
from datetime import datetime

#Stocks that you want to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA", "FB", "AAPL"]

#How far back to gather data?
year = 2015
month = 1
day = 1

#How many days out to predict 
forecast_out = 10

#Path to save file directory
save_path = r'home/mason/Capstone/Figures/'

#Path to data directory
path = r'/home/mason/Capstone/Data/'


#Sets start date for stock datetime
today = date.today()
start = date(year, month, day)
start = start.strftime("%Y-%m-%d")

#Sets end date for stock datetime
end = today.strftime("%Y-%m-%d")


now = datetime.now()

print(now)




