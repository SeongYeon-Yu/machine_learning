#기계학습응용 8주차과제
#60181912 유성연
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def make_toyimg():
    image = mpimg.imread("/Users/lg/Downloads/edge_detection_ex.jpg")
    image.astype(np.float)
    print(image.shape)
    plt.imshow(image, cmap='Greys')
    plt.show()
    # data format should be change to  batch_shape + [height, width, channels].
    image = image.reshape((1, 720, 1280, 3))
    image = tf.constant(image, dtype=tf.float64)
    return image

def make_toyfilter():
    weight = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1],[-1, -2, -1],[0, 0, 0],[1, 2, 1],[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    weight=weight.reshape((1,3,3,3))
    print(weight.shape)
    print("weight.shape", weight.shape)
    weight_init = tf.constant_initializer(weight)

    return weight_init

def cnn_valid(image, weight):
    conv2d = keras.layers.Conv2D(filters=1, kernel_size=3,
                                 kernel_initializer=weight)(image)
    print("conv2d.shape", conv2d.shape)
    print(conv2d.numpy())
    plt.imshow(conv2d.numpy().reshape(-1,1278), cmap='gray')
    plt.show()

def main():
    img = make_toyimg()
    filter = make_toyfilter()
    cnn_valid(img, filter)

main()
