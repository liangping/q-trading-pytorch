# -*- coding: utf-8 -*-
"""StockPriceLSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/nandahkrishna/StockTrading/blob/master/ForecastingLSTM.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import numpy as np
import pandas as pd 
from tqdm._tqdm_notebook import tqdm_notebook
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# %matplotlib inline

aapl = pd.read_csv("Dataset/Stocks/aapl.us.txt", sep=",").iloc[5000:]
aapl = aapl.drop(["Date", "OpenInt"], axis=1)
aapl = aapl[["Open", "High", "Low", "Volume", "Close"]]

aapl.tail()

plt.plot(aapl["Close"])
plt.show()

train, test = train_test_split(aapl, train_size=0.8, test_size=0.2, shuffle=False)

scaler, scaler_vol = MinMaxScaler(), MinMaxScaler()
train["Open"] = scaler.fit_transform(np.array(train["Open"]).reshape(-1, 1))
train["High"] = scaler.transform(np.array(train["High"]).reshape(-1, 1))
train["Low"] = scaler.transform(np.array(train["Low"]).reshape(-1, 1))
train["Close"] = scaler.transform(np.array(train["Close"]).reshape(-1, 1))
test["Open"] = scaler.transform(np.array(test["Open"]).reshape(-1, 1))
test["High"] = scaler.transform(np.array(test["High"]).reshape(-1, 1))
test["Low"] = scaler.transform(np.array(test["Low"]).reshape(-1, 1))
test["Close"] = scaler.transform(np.array(test["Close"]).reshape(-1, 1))
train["Volume"] = scaler_vol.fit_transform(np.array(train["Volume"]).reshape(-1, 1))
test["Volume"] = scaler_vol.transform(np.array(test["Volume"]).reshape(-1, 1))

train.head()

test.head()

params = {
    "batch_size": 10,
    "epochs": 150,
    "lr": 0.0001,
    "time_steps": 60
}

def preprocess(data):
    rows = data.shape[0] - params["time_steps"]
    cols = data.shape[1]
    x = np.zeros((rows, params["time_steps"], cols))
    y = np.zeros((rows,))
    for i in tqdm_notebook(range(rows)):
        x[i] = data[i: params["time_steps"] + i]
        y[i] = data[params["time_steps"] + i, 4]
    print(x.shape, y.shape)
    return x, y

x_train, y_train = preprocess(np.array(train))

x_train = x_train[:-1]
y_train = y_train[:-1]

model = Sequential()
model.add(LSTM(40, batch_input_shape=(params["batch_size"], params["time_steps"], x_train.shape[2]),
               stateful=True, return_sequences=True, dropout=0.0, recurrent_dropout=0.0,
               kernel_initializer="random_uniform"))
model.add(Dropout(0.2))
model.add(LSTM(40, batch_input_shape=(params["batch_size"], params["time_steps"], x_train.shape[2]),
               stateful=True, return_sequences=True, dropout=0.0, recurrent_dropout=0.0,
               kernel_initializer="random_uniform"))
model.add(Dropout(0.2))
model.add(LSTM(40, batch_input_shape=(params["batch_size"], params["time_steps"], x_train.shape[2]),
               stateful=True, dropout=0.0, recurrent_dropout=0.0, kernel_initializer="random_uniform"))
model.add(Dropout(0.2))
model.add(Dense(20, activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="adam")

model.summary()

x_temp, y_temp = preprocess(np.array(test))
x_temp, y_temp = x_temp[:-1], y_temp[:-1]

x_val, x_test_t = np.split(x_temp, 2)
y_val, y_test_t = np.split(y_temp, 2)

x_val = x_val[:-6]
y_val = y_val[:-6]
x_test_t = x_test_t[:-6]
y_test_t = y_test_t[:-6]

es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=40, min_delta=0.0001)
mc = ModelCheckpoint("best_model.h5", monitor="val_loss", verbose=1, save_best_only=True,
                     save_weights_only=False, mode='min', period=1)

history = model.fit(x_train, y_train, epochs=params["epochs"], verbose=2, batch_size=params["batch_size"],
                    shuffle=False, validation_data=(x_val, y_val), callbacks=[es, mc])

pickle.dump(model, open("lstm_model.pkl", "wb"))

y_pred = model.predict(x_test_t, batch_size=params["batch_size"])

y_actual = scaler.inverse_transform(y_pred)

y_true = scaler.inverse_transform(y_test_t.reshape(-1, 1))

mean_squared_error(y_true, y_actual)

plt.figure()
plt.plot(y_actual)
plt.plot(y_true)
plt.show()