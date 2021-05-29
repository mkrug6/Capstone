import sys
sys.path.append("/home/mason/Capstone/")
import matplotlib.pyplot as plt

def model_graph(rbf_svr, days, close_prices, tick, save=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        plt.savefig("./Capstone/Figures/Prediction_" + tick + '.png')
    return plt.show()



def predict_graph(rbf_svr, days, close_prices, future_array, future_close_prices, all_days, all_close_prices, tick, save=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        plt.savefig("./Capstone/Figures/Prediction_" + tick + '.png')
    return plt.show()





import os
cwd = os.getcwd()

print(cwd)
