import yfinance as yf

def download_csv(start, end, ticker, path):
    #Uses ticker[] to iteratively download CSVs
    for i in ticker:
        stock = []
        stock = yf.download(i,start=start, end=end)
        stock.to_csv (path + i + '.csv', index = True, header=True)
    return