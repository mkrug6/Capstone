import yfinance as yf
from datetime import date
from config import *
import pandas as pd
import os

print(path)
print(config.path)

#Sets start date for stock datetime
today = date.today()
start = date(2020,1,1)
start = start.strftime("%Y-%m-%d")
print(start)

#Sets end date for stock datetime
end = today.strftime("%Y-%m-%d")
print(end)


#Tickers to download
ticker = ["SPY", "GOOG", "MSFT", "TSLA"]
print (ticker)


#Empty df to iterate over
stock = []

#Fill the df with date

for i in ticker:
    stock = yf.download(i,start=start, end=end)
    stock.to_csv (path + i + '.csv', index = True, header=True)













p = Path('./')
input_file.to_csv(Path(p, 'match_' + filename + '.csv')), index=False)



"""
for i in ticker:
    print(i)

    stock = []
    stock = yf.download(i,start=start, end=end)
    stock.head()
"""













#def download_csv(ticker, date_start):
#    data = yf.download(ticker, date_start)
#
#   return data
