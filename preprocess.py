import pandas as pd
#from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import sys
#import os
#import pickle
#from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
#from feature_format import featureFormat, targetFeatureSplit
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn import svm, datasets
#from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


# Load the data into a pandas dataframe

df = pd.read_csv(r'./Capstone/Data/SPY.csv')

# Make the dates the index

df.index = df['Date']

# Remove all but close column
# aka create the target feature (what we are interested in predictin)

target = df['Close']

#Also known as big X
features = df.drop(['Close', 'Date'], axis=1)

# Assemble the other features



# Process it to prepare for scaling

df = data.reshape(-1, 1)

target = data.reshape(-1, 1)
features = data.reshape(-1, 1)
#Preprocess that data so it's easier to work with




from sklearn.svm import SVR
Regressor_SVR = SVR(kernel='rbf')



























print(features.shape)
print(target.shape)



scaler = MinMaxScaler()

df = scaler.fit_transform(df)

# Make a plot of it just for funsies

plt.plot(df)
plt.show()


# This is where I might go to a new file



# Start SVM classifier
# What is better than train test split? Probably gridsearch




x_train, y_train, x_test, y_test =  train_test_split(features, target, test_size=0.3, random_state=42)


print(x_train.shape)
print(y_train.shape)



#Import svm model


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


#####From the video
##### https://www.youtube.com/watch?v=Iu5T2TZumxU&ab_channel=ComputerScienceComputerScience


actual_price = df.tail(1)

df = df.head(len(df)-1)

days = list()

close_prices = list()

df_days = df.loc[:, 'Date']

df_close_price = df.loc[:, 'Close']

#create independent data set (the dates/x)

for day in df_days:
    days.append([int(day.split('-')[2])])

#create dependent data set (close price/y)

for close_price in df_close_price:
    close_prices.append(float(close_price))



lin_svr  = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, close_prices)


poly_svr  = SVR(kernel='poly', C=1000.0)
poly_svr.fit(days, close_prices)

rbf_svr  = SVR(kernel='rbf', C=1000.0, gamma=0.85)
rbf_svr.fit(days, close_prices)























#####End from the video








def evaluate_clf(grid_search, features, labels, params, iters=100):
    """
    Evaluate a classifier
    """
    acc = []
    pre = []
    recall = []

    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)]
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))