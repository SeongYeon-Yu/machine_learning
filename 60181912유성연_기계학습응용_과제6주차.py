#60181912 유성연
#기계학습응용 6주차과제

import numpy as np
import tensorflow as tf
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def makemodel(x_train, y_train, x_valid, y_valid):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[1, 64]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x_train, y_train, epochs=30,
                        validation_data=(x_valid, y_valid), callbacks=[tb_hist])
    return model,history


def main():
    digits = load_digits()
    x_data = digits.data
    y_data = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    x_valid, x_train = x_train[:300], x_train[300:]
    y_valid, y_train = y_train[:300], y_train[300:]
    makemodel(x_train, y_train, x_valid, y_valid)

    n_rows = 1
    n_cols = 10
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(x_test[index].reshape(8, 8), cmap="binary", interpolation="nearest")
            plt.axis('off')

    plt.show()
main()
