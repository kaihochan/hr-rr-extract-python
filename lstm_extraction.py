import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation

import os

def get_data():
    df = pd.read_excel('./Data/Lay_Up_without_anything_Test_1_John.xlsx', header=None, usecols='B')
    data = df.values
    
    # normalise the data in a feature scale of [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

    # split into train and test dataset
    train_size = int(round(0.9 * len(data)))
    train, test = data[:train_size, :], data[train_size:, :]
    x_train, y_train = create_dataset(train)
    x_test, y_test = create_dataset(test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test

def create_dataset(dataset):
    x, y = [], []
    for i in np.arange(len(dataset)-2):
        x.append(dataset[i:(i+1), 0])
        y.append(dataset[i+1, 0])
    return np.array(x), np.array(y)

def get_model() -> Sequential:
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def predict_signal(model, x_test):
    predict = model.predict(x_test)
    predict = np.reshape(predict, (predict.size,))
    return predict

def plot_graph(y_test, predict) -> None:
    plt.plot(y_test, 'b', label="True Data")    
    plt.plot(predict, 'r', label="Prediction")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data()
    model = get_model()
    model.fit(x_train, y_train, batch_size=512, epochs=1, validation_split=0.05)
    predict = predict_signal(model, x_test)
    plot_graph(y_test, predict)

