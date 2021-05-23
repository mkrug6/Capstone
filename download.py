import yfinance as yf
from datetime import date
import os

#Stocks that you want to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA", "FB", "AAPL"]

#Path to save file directory
path = r'./Data/'

#Sets start date for stock datetime
today = date.today()
start = date(2020,1,1)
start = start.strftime("%Y-%m-%d")

#Sets end date for stock datetime
end = today.strftime("%Y-%m-%d")

#Fill the df with date. Takes in ticker as a dictionary/list
def download_csv(start, end, ticker):
    for i in ticker:
        stock = []
        stock = yf.download(i,start=start, end=end)
        stock.to_csv (path + i + '.csv', index = True, header=False)
    return

download_csv(start, end, ticker)
