import time
import warnings
import math
import pandas as pd
import numpy as np
from numpy import newaxis
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


np.random.seed(1234)
warnings.filterwarnings("ignore")

def create_dataset(data, look_back):
    result = []
    for index in range(len(data) - look_back):
        result.append(data[index:index+look_back])
    result = np.array(result)

    result_mean = result.mean()
    result -= result_mean

    row = round(result.shape[0]) 
    row = int(row)
    # First 100 entries for testing and rest for training
    train = result[:row-100, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row-100:row, :-1]
    y_test = result[row-100:row, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [X_train, y_train, X_test, y_test]

def build_model():
    model = Sequential()
    layers = [1, 4, 8, 16, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(LSTM(
        layers[3],
        return_sequences=False))
    #model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[4]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


look_back = 4
epochs = 100

dataset = pd.read_csv("aapl1.csv", parse_dates = {'dateTime': ['Date']}, index_col = 'dateTime')
dataset.sort_index(axis=0, inplace=True)

data = dataset['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

X_train, y_train, X_test, y_test = create_dataset(data, look_back)

model = build_model()

model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.08)
predicted = model.predict(X_test)
trainPredicted = model.predict(X_train)
testPredicted = model.predict(X_test)
predicted = np.reshape(testPredicted, (testPredicted.size,))
trainScore = math.sqrt(mean_squared_error(scaler.inverse_transform(y_train), scaler.inverse_transform(trainPredicted)))
testScore = math.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(testPredicted)))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

# Necessary (along with enabling X11 forwarding) if remote login to Unix systems from Windows
plt.switch_backend('TkAgg')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(scaler.inverse_transform(y_test[:100]))
plt.plot(scaler.inverse_transform(predicted[:100]))
plt.show()
