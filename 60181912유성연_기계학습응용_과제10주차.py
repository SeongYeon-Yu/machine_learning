#60181912유성연
#기계학습응용 10주차과제
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

def load_data():
    # 먼저 MNIST 데이터셋을 로드하겠습니다. 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다. 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다. 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # data를 가져오겠다
    X_train_full = X_train_full.astype(np.float32) #x_train_full을 float형태로 바꾸겠다.
    X_test = X_test.astype(np.float32) # x_test을 float형태로 바꾸겠다
    return X_train_full, y_train_full, X_test, y_test
    # 값 반환하겠다

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255 # 입력데이터 0에서 1사이로 값 바꾸기 위해 255로 나눠 정규화하겠다

    X_test = X_test / 255. #255로 나눠서 정규화
    train_feature = np.expand_dims(X_train_full, axis=3)  # x_train_full의 차원을 3으로 확장하여 train_feature라고 하겠다.
    test_feature = np.expand_dims(X_test, axis=3) # x_test의 차원을 3으로 확장하여  test_feature라고 하겠다.

    print(train_feature.shape, train_feature.shape) # train_feature의 형 출력
    print(test_feature.shape, test_feature.shape)# test_feature의 형 출력

    return train_feature,  test_feature # train_feature, test_feature 반환


def draw_digit(num): #그림 그리기
    for i in num:
        for j in i:
            if j == 0:
                print('0', end='')
            else :
                print('1', end='')
        print()

def makemodel(X_train, y_train, X_valid, y_valid, weight_init): #모델 만들기
    model = Sequential() # 모델 순차적
    model.add(Conv2D(32, kernel_size=(3, 3),  activation='relu')) # 커널사이즈 3x3, relu하여 커널갯수 32개 = n1
    model.add(MaxPooling2D(pool_size=2)) # 풀링사이즈2인 큰값(max)으로 풀링 하겠다
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # n2 = 64
    model.add(MaxPooling2D(pool_size=2)) # 큰 값으로 풀링 하겠다.
    model.add(Dropout(0.25)) # 과적합되는것을 막기위해 25퍼센트 dropout
    model.add(Flatten()) #flatten
    model.add(Dense(128, activation='relu')) # 128개의 relu함수 있는 히든레이어
    model.add(Dense(10, activation='softmax')) #0~9까지로 10개의 라벨 소프트맥스

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # 모델 컴파일


    return model #model 반환

def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16,10)) #가로 16, 세로 10 사이즈의 그림으로 그리겠다

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val') # title로 그림의 값을 쓰겠다.
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train') #title로 값을 말예측한 값을 쓰겠다.

    plt.xlabel('Epochs') #xlabel에 epochs라고 쓰겠다.
    plt.ylabel(key.replace('_',' ').title()) #ylabel에 key_replace의 타이틀을 쓰겠따 (accuracy)
    plt.legend() # 선별로 무엇인지 표시하겠다.

    plt.xlim([0,max(history.epoch)])#0부터 , 에폭의 최대값까지 x의 범위를 지정하겠다.
    plt.show()#그림 그리기



def draw_prediction(pred, k,X_test,y_test,yhat): #예측
    samples = random.choices(population=pred, k=16)
    #랜덤으로 선택하겠다
    count = 0#0대입
    nrows = ncols = 4  # 4대입
    plt.figure(figsize=(12,8))  #그림그리기

    for n in samples:
        count += 1 #1증가
        plt.subplot(nrows, ncols, count) # 작은 네모의 subplot를 괄호안의 수만큼만들겠다
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
        # 이미지를 보이겠다
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n]) #tmp를 Label에 테스트한값, Prediction으로 예측한값이라 하겠다.
        plt.title(tmp) #타이틀에 위에 정의한 tmp라고 쓰겠다.

    plt.tight_layout() # 레이아웃 타이트하게 하겠다.
    plt.show() #이미지 보이겠다.

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test) #x_test로 모델 예측
    yhat = yhat.argmax(axis=1) #가장 큰값을 yhat에 넣겠다

    print(yhat.shape) #yhat으ㅣ 크기 출력
    answer_list = []  #answer_list에 리스트 만들겠다

    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]: #일치하면
            answer_list.append(n) #리스에 추가하겠다

    draw_prediction(answer_list, 16,X_test,y_test,yhat)
    # 리스트 예측하여 그림
    answer_list = []

    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]: # 일치하지않으면
            answer_list.append(n) #리스트에 추가

    draw_prediction(answer_list, 16,X_test,y_test,yhat)
    #리스트 예측후 그리기
def main():
    X_train, y_train, X_test, y_test = load_data() # 데이터 가져오겠따
    X_train, X_test = data_normalization(X_train,  X_test) #데이터 정규화

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform')
    #model을 makemodel함수를 호출하여 glort_uniform으로 초기화하겠다.


    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=2,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2)
    # 모델 핏하겠다. verbose=2중간에있는 값 보여주겠다

    evalmodel(X_test, y_test, model) #모델 test하겠다
    plot_history([('baseline', baseline_history)]) # 정확도 그림 그리기

main()
