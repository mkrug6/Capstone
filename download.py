import yfinance as yf

def download_csv(start, end, path, i):
    stock = []
    stock = yf.download(i,start=start, end=end)
    stock.to_csv (path + i + '.csv', index = True, header=True)
    return