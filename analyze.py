import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import config


plt.style.use('bmh')

df = pd.read_csv('./Data/SPY.csv')
df.head(5)

#get number of trading days
df.shape

#visualize close Price
plt.figure(figsize=(16,8))
plt.title('SPY')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(df['Close'])
plt.show()

df = df[['Close']]
df.head(6)

#predict x days into the future

future_days = 25

#create a new column (target) shifted x units/days up

df['Prediction'] = df[['Close']].shift(-future_days)
df.head(4)
df.tail(4)

#Create the feature data set (X) and convert to np array and remove 'x' rows/Days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)

#create target data set (y) and convert to np array and get all of the target values except the last 'x' rows/Days

y = np.array(df['Prediction'])[:-future_days]
print(y)

#Split data into 75% train, 25% test_size

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#create the model_selection
#create decision tree regressor models
tree = DecisionTreeRegressor().fit(x_train, y_train)

#create linear regression model

lr = LinearRegression().fit(x_train, y_train)

#get last 'x' rows from feature datasets
x_future = df.drop(['Prediction'], 1)[:-future_days]

#get the last 'x' rows of feature datasets

x_future = x_future.tail(future_days)

x_future = np.array(x_future)

x_future

#show the model tree Prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)

print()
#show mode lr Prediction

lr_prediction = lr.predict(x_future)
print(lr_prediction)

#visualize the dataset

predictions = tree_prediction

valid = df[X.shape[0]:]

valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Original', 'Val', 'Pred'])
plt.show()


# lr predictions

predictions = lr_prediction

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Original', 'Val', 'Pred'])

plt.show()



#------------------------------------------------------
#Older code to maybe incorporate



def try_classifiers(data, features_list):
    """
    Tries different classifiers and then chooses the best one
    """

    data = featureFormat(data, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    print('Trying AdaBoost')
    clf_ab = AdaBoostClassifier(DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=2,
        class_weight='balanced'),
        n_estimators=50,
        learning_rate=.8)

    clf_ab_grid_search = GridSearchCV(clf_ab, {})
    clf_ab_grid_search.fit(features_train, labels_train)
    clf_ab_grid_search.best_estimator_
    test_classifier(clf_ab_grid_search, data, features_list)

    print('Trying GaussianNB')
    clf_gb = GaussianNB()
    clf_gb_grid_search = GridSearchCV(clf_gb, {})
    clf_gb_grid_search.fit(features_train, labels_train)
    clf_gb_grid_search.best_estimator_

    print('Trying SVC')

    clf_svc = SVC(kernel='linear', max_iter=1000)
    clf_svc_grid_search = GridSearchCV(clf_svc, {})
    clf_svc_grid_search.fit(features_train, labels_train)
    clf_svc_grid_search.best_estimator_

    # Return the one which perform the best
    return clf_ab_grid_search



































"""
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split




#DataFlair - Get Amazon stock data
amazon = quandl.get("WIKI/AMZN")
print(amazon.head())



#DataFlair - Get only the data for the Adjusted Close column
amazon = amazon[['Adj. Close']]
print(amazon.head())



forecast_len=30
amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_len)
print(amazon.tail())


#DataFlair - Drop the Predicted column, turn it into a NumPy array to create dataset
x=np.array(amazon.drop(['Predicted'],1))
#DataFlair - Remove last 30 rows
x=x[:-forecast_len]
print(x)


#DataFlair - Create dependent dataset for predicted values, remove the last 30 rows
y=np.array(amazon['Predicted'])
y=y[:-forecast_len]
print(y)

#DataFlair - Split datasets into training and test sets (80% and 20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#DataFlair - Create SVR model and train it
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_train,y_train)


#DataFlair - Get score
svr_rbf_confidence=svr_rbf.score(x_test,y_test)
print(f"SVR Confidence: {round(svr_rbf_confidence*100,2)}%")

#DataFlair - Create Linear Regression model and train it
lr=LinearRegression()
lr.fit(x_train,y_train)


#DataFlair - Get score for Linear Regression
lr_confidence=lr.score(x_test,y_test)
print(f"Linear Regression Confidence: {round(lr_confidence*100,2)}%")











#importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
#used for setting the output figure size
rcParams['figure.figsize'] = 20,10
#to normalize the given input data
scaler = MinMaxScaler(feature_range=(0, 1))
#to read input data set (place the file name inside  ' ') as shown below
todataframe = pd.read_csv('./Data/MSFT.csv')
#to print the first few data in the data set
todataframe.head()


todataframe['Date'] = pd.to_datetime(todataframe.Date,format='%Y-%m-%d')
todataframe.index = todataframe['Date']
plt.figure(figsize=(16,8))
plt.plot(todataframe['Close'], label='Closing Price')

#importing the packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#dataframe creation
seriesdata = todataframe.sort_index(ascending=True, axis=0)
new_seriesdata = pd.DataFrame(ind(todataframe)),columns=['Date','Close'])
length_of_data=len(seriesdata)
for i in range(0,length_of_data):
    new_seriesdata['Date'][i] = seriesdata['Date'][i]
    new_seriesdata['Close'][i] = seriesdata['Close'][i]
#setting the index again
new_seriesdata.index = new_seriesdata.Date
new_seriesdata.drop('Date', axis=1, inplace=True)
#creating train and test sets this comprises the entire data’s present in the dataset
myseriesdataset = new_seriesdata.values
totrain = myseriesdataset[0:255,:]
tovalid = myseriesdataset[255:,:]
#converting dataset into x_train and y_train
scalerdata = MinMaxScaler(feature_range=(0, 1))
scale_data = scalerdata.fit_transform(myseriesdataset)
x_totrain, y_totrain = [], []
length_of_totrain=len(totrain)
for i in range(60,length_of_totrain):
    x_totrain.append(scale_data[i-60:i,0])
    y_totrain.append(scale_data[i,0])
x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
x_totrain = np.reshape(x_totrain, (x_totrain.shape[0],x_totrain.shape[1],1))
#LSTM neural network
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_totrain.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adadelta')
lstm_model.fit(x_totrain, y_totrain, epochs=3, batch_size=1, verbose=2)
#predicting next data stock price
myinputs = new_seriesdata[len(new_seriesdata) - (len(tovalid)+1) - 60:].values
myinputs = myinputs.reshape(-1,1)
myinputs  = scalerdata.transform(myinputs)
tostore_test_result = []
for i in range(60,myinputs.shape[0]):
    tostore_test_result.append(myinputs[i-60:i,0])
tostore_test_result = np.array(tostore_test_result)
tostore_test_result = np.reshape(tostore_test_result,(tostore_test_result.shape[0],tostore_test_result.shape[1],1))
myclosing_priceresult = lstm_model.predict(tostore_test_result)
myclosing_priceresult = scalerdata.inverse_transform(myclosing_priceresult)

"""
