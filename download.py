#import sys, os
#sys.path.append('/home/mason/Capstone/')

import yfinance as yf
#from datetime import date
from config_file import *

#Fill the df with date. Takes in ticker as a dictionary/list
def download_csv(start, end, ticker):
    for i in ticker:
        stock = []
        stock = yf.download(i,start=start, end=end)
        stock.to_csv (path + i + '.csv', index = False, header=False)
    return

download_csv(start, end, ticker)
