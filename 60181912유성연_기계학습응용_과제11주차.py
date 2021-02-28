#60181912 유성연
#기계학습응용 11주차과제
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
# convert into dataset matrix

def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data)- step):
        d =i + step
        X.append(data[i:d, ])
        Y.append(data[d,])
    return np.array(X), np.array(Y)

step =4
N = 1000
Tp = 800
t = np.arange(0, N)
x = np.sin(0.02 * t) + 2 * np.random.rand(N)
df = pd.DataFrame(x)
df.head()

#plt.plot(df)
#plt.show()

values = df.values
train, test = values[0:Tp, :], values[Tp:N, :]
print(train.shape)

# 맨마지막의 데이터에 추가 적으로 복사로 더 데이터 넣기
test = np.append(test, np.repeat(test[-1,], step))
train = np.append(train, np.repeat(train[-1,], step))

# 데이터 자르기
trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# 데이터를 4개씩 잘라서 reshape
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(SimpleRNN(50,  return_sequences=False, input_shape=(1,4)))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(trainX,trainY,epochs=200,  verbose=2)
train = model.predict(trainX)
test= model.predict(testX)
predicted=np.concatenate((train,test),axis=0)

index = df.index.values
plt.plot(df)
plt.plot(index,predicted)
plt.axvline(df.index[Tp], c="r")
plt.show()