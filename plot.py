import sys
sys.path.append("/home/mason/Capstone/")
import matplotlib.pyplot as plt

def model_graph(rbf_svr, days, close_prices, tick, save=bool, show=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        plt.savefig("/home/mason/Capstone/Figures/Prediction_" + tick + '.png')
    if show:
        plt.show()
    return 



def predict_graph(rbf_svr, days, close_prices, future_array, future_close_prices, all_days, all_close_prices, tick, save=bool, show=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        plt.savefig("/home/mason/Capstone/Figures/Prediction_" + tick + '.png')
    if show:
        plt.show()
    return

#save_path = 'r/home/mason/Capstone/Figures/'