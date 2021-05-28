def model_graph(days, close_prices, save=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        print("Put save code here")
    return plt.show()



def predict_graph(days, close_prices, save=bool):
    # Graphs actual close prices against models predicted valeus    
    
    plt.figure(figsize=(32, 16))
    plt.scatter(days, close_prices, color='black', label='Original Data')
    plt.plot(days, rbf_svr.predict(days), color='blue', label='RBF Model')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Support Vector Regression')
    #input a legend here plt.legend()
    if save:
        print("Put save code here")
    return plt.show()