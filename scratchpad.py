


#-----------------------Split df into array of features, target

# Make the features and make into np array
features = np.array(df.drop(['Close', 'Date'], axis=1))
x = features
# Make the target and turn in np array
target = np.array(df['Close'])
y = target

# Data shifted for prediction purposes
X = x[:-forecast_out]

#-----------------------Create train/test data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, train_size=0.75, random_state=76)

#-----------------------Scale the data for uniformity/normalization

scaler = MinMaxScaler(feature_range=(0, 1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Must reshape y_train before scaling
y_train = y_train.reshape(-1, 1)

y_train = scaler.fit_transform(y_train)






#-----------------------SVM 

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel='rbf')

Regressor_SVR.fit(X_train, y_train.ravel())

Predicted_values_SVR = Regressor_SVR.predict(X_test)

Predicted_values_SVR = Predicted_values_SVR.reshape(-1,1)

Predicted_values_SVR = scaler.inverse_transform(Predicted_values_SVR)











import matplotlib.dates as mdates

# change the dates into ints for training 
dates_df = df.copy()
dates_df = dates_df.reset_index()

# Store the original dates for plotting the predicitons
org_dates = dates_df['Date']

# convert to ints
dates_df['Date'] = dates_df['Date'].map(mdates.date2num)

dates_df.tail()





















































#---------------------------------------Actual Prediction






#---------------------------------------Plot the stuff


  
# Assign variables to the y axis part of the curve
xchart = df['Date']
ychart = df['Close']
zchart = Predicted_values_SVR
  
# Plotting both the curves simultaneously
plt.plot(xchart, ychart, color='r', label='sin')
plt.plot(xchart, zchart, color='g', label='cos')
  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Angle")
plt.ylabel("Magnitude")
plt.title("Sine and Cosine functions")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()














#---------------------------------Metrics to Measure Performance

MAE = mean_absolute_error(y_test, Predicted_values_SVR)

print(MAE)

MSE = mean_squared_error(y_test, Predicted_values_SVR)

print(MSE)

RMSE = math.sqrt(MSE)

print(RMSE)

R2 = r2_score(y_test, Predicted_values_SVR)

print(R2)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred)/(y_true))*100

MAPE = mean_absolute_percentage_error(y_test, Predicted_values_SVR)                                                 
print(MAPE)






##########Probably don't need this model






#---------------------------------Apply the Linear Regression Model

Linear_R = LinearRegression()

Linear_R.fit(X_train, y_train)

scaler = MinMaxScaler(feature_range=(0, 1))

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

#Must reshape y_train before scaling

y_train = y_train.reshape(-1, 1)

y_train = scaler.fit_transform(y_train)

Linear_R = LinearRegression()

Linear_R.fit(X_train, y_train)

Predicted_values_MLR = Linear_R.predict(X_test)

Predicted_values_MLR = scaler.inverse_transform(Predicted_values_MLR)









