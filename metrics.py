from config import ticker, metrics_dict, deviation_dict

def generate_metrics(acp):
    for i in range(0, len(ticker), 1):
        name = ticker[i]
        x = acp[name]
        y = metrics_dict[name]
        deviation = ((abs(x - y)) / x) * 100
        deviation = round(deviation, 2)
        deviation_dict[name] = deviation
        deviation = str(deviation)
        print('The deviation for ' + name + ' is: ' + deviation + '%.')
    return

def average_deviation():
    a = 0
    for i in range(0, len(ticker), 1):
        name = ticker[i]
        a = a + deviation_dict[name]
    a = a/len(ticker)
    a = round(a, 2)
    print()
    print('The average deviation between all ' + str(len(ticker)) + ' stocks is: ' + str(a) + '.')
    
    
    

