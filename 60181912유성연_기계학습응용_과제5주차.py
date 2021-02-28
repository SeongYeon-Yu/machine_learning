#60181912 유성연
#기계학습응용 5주차
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

x1 = np.random.randint(101,size=(100,3))
x2 = np.random.randint(50,101,size=(100,3))
X =np.concatenate([x1,x2]).reshape(200,3)

y1 = np.zeros((100,1))
y2 = np.ones((100,1))
Y=np.concatenate([y1,y2]).reshape(200,1).ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train,Y_train)
pred=model.predict((X_test))
prob = accuracy_score(Y_test,pred)
print(pred)
print(prob)
