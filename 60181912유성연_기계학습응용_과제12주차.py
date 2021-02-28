#기계학습응용 12주차
# 60181912유성연
import numpy as np
from tensorflow.keras import Sequential, Model
from keras.layers import SimpleRNN,TimeDistributed,Embedding, Dense, Activation


X = []
Y = []
for i in range(1,5,1):
    lst = list(range(i,i+3))
    X.append(list(map(lambda c: [c/10], lst)))
    lst1 = list(range(i+1,i+4))
    Y.append(list(map(lambda c: [c/10], lst1)))
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)

model = Sequential()
model.add(SimpleRNN(100,  return_sequences=True, input_shape=(3,1)))

model.add(TimeDistributed(Dense(1)))
model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X,Y,epochs=200,  verbose=2)

X_test = np.array([[[0.5],[0.6],[0.7]]])
print(model.predict(X_test))
X_test = np.array([[[0.6],[0.7],[0.8]]])
print(model.predict(X_test))


model1 = Sequential()
model1.add(SimpleRNN(100,  return_sequences=True, input_shape=(3,1)))
model1.add(SimpleRNN(100, return_sequences=True))

model1.add(TimeDistributed(Dense(1)))
model1.summary()

model1.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model1.fit(X,Y,epochs=200,  verbose=2)

X_test = np.array([[[0.5],[0.6],[0.7]]])
print(model1.predict(X_test))
X_test = np.array([[[0.6],[0.7],[0.8]]])
print(model1.predict(X_test))
