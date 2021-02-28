#60181912 유성연
#기계학습응용 7주차과제

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
import time

def dense(label_dim, weight_init, activation):
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init, activation=activation)

def makemodel(x_train, y_train, x_test, y_test):
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[1, 64]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    model.summary()
    return model


def makemodeldrop(x_train, y_train, x_test, y_test, weight_init):
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[1, 64]))
    model.add(dense(300, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(100, weight_init, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(dense(10, weight_init, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    model.summary()
    return model

def modelpredict(model, x_train, y_train, x_valid, y_valid):
    # 시간 측정
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    start = time.time()
    history = model.fit(x_train, y_train, epochs=30,
                        validation_data=(x_valid, y_valid), callbacks=[tb_hist])
    print("time :", time.time() - start)
    return history


def plot_history(histories, key='accuracy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.show()


def main():
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    model_RandomNormal = makemodel(x_train, y_train, x_test, y_test)
    model_xavier = makemodeldrop(x_train, y_train, x_test, y_test, 'glorot_uniform')

    hist_xavier = modelpredict(model_xavier, x_train, y_train, x_test, y_test)
    hist_RandomNormal = modelpredict(model_RandomNormal, x_train, y_train, x_test, y_test)

    plot_history([('Normal', hist_RandomNormal), ('Dropout', hist_xavier)])


main()
