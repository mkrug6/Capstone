#import sys, os
#sys.path.append('/home/mason/Capstone/')

import yfinance as yf
#from datetime import date
from config import *

#Fill the df with date. Takes in ticker as a dictionary/list
def download_csv(start, end, ticker):
    for i in ticker:
        stock = []
        stock = yf.download(i,start=start, end=end)
        stock.to_csv (path + i + '.csv', index = True, header=True)
    return

download_csv(start, end, ticker)




#remove when making main.py
import sys
sys.path.append("/home/mason/Capstone/")
