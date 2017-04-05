import numpy as np
import math
import warnings
import datetime
import time
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

df = pd.read_csv("aapl1.csv", parse_dates = {'dateTime': ['Date']}, index_col = 'dateTime')

df.sort_index(axis=0, inplace=True)
#df['Close'].plot()
#plt.show()

X = df[['Open', 'Adj Close']]
y = df['Close']

start_test = datetime.datetime(2016, 11, 4)

X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]

scalerX = StandardScaler().fit(X_train)
scalery = StandardScaler().fit(y_train)
X_train = scalerX.transform(X_train)
y_train = scalery.transform(y_train)
X_test = scalerX.transform(X_test)
y_test = scalery.transform(y_test)

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
	clf.fit(X_train, y_train)
	#print("Coefficient of determination on training set:",clf.score(X_train, y_train))
	# create a k-fold cross validation iterator of k=5 folds
	cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
	scores = cross_val_score(clf, X_train, y_train, cv=cv)
	#print("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
	trainPredicted = clf.predict(X_train)
	predicted = clf.predict(X_test)
	trainScore = math.sqrt(mean_squared_error(scalery.inverse_transform(y_train), scalery.inverse_transform(trainPredicted)))
	testScore = math.sqrt(mean_squared_error(scalery.inverse_transform(y_test), scalery.inverse_transform(predicted)))
	print('Train Score: %.2f RMSE' % (trainScore))
	print('Test Score: %.2f RMSE' % (testScore))
	plt.plot(scalery.inverse_transform(y_test))
	plt.plot(scalery.inverse_transform(predicted))
	plt.show()

def train_linear(reg, X_train, y_train, X_test, y_test):	
	reg.fit(X_train,y_train)
	trainPredicted = reg.predict(X_train)
	testPredicted = reg.predict(X_test)

	trainScore = math.sqrt(mean_squared_error(scalery.inverse_transform(y_train), scalery.inverse_transform(trainPredicted)))
	testScore = math.sqrt(mean_squared_error(scalery.inverse_transform(y_test), scalery.inverse_transform(testPredicted)))
	print('Train Score: %.2f RMSE' % (trainScore))
	print('Test Score: %.2f RMSE' % (testScore))
	plt.plot(scalery.inverse_transform(y_test))
	plt.plot(scalery.inverse_transform(testPredicted))
	plt.show()

reg = linear_model.LinearRegression()
train_linear(reg, X_train, y_train, X_test, y_test)

clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='l2',  random_state=42)
train_and_evaluate(clf_sgd,X_train,y_train, X_test, y_test)

clf_svr_poly = svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train, X_test, y_test)

clf_svr_rbf = svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train, X_test, y_test)


