#60181912 유성연
#기계학습응용 13주차

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import SimpleRNN, TimeDistributed, Embedding, Dense, LSTM

df_price = pd.read_csv('C:/Users/lg/Desktop/hw.csv')

seq_len = 50 #window 값이 50
sequence_length = seq_len + 1
high_prices = df_price['고가'].values
low_prices = df_price['저가'].values
mid_prices = (high_prices + low_prices)/2
result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index:index + sequence_length])
normalized_data = []
for window in result:
    normalized_window = [[(float(p) / float(window[0]))-1]for p in window]
    normalized_data.append(normalized_window)
result = np.array(normalized_data)
x_train = result[:8000,:-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = result[:8000, -1]
x_test = result[8000:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
y_test = result[8000:, -1]

model = Sequential()

model.add(LSTM(50, return_sequences = True, input_shape=(50,1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')


model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=50)
pred = model.predict(x_test)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
