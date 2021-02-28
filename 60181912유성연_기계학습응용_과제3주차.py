#60181912 유성연
#기계학습응용 과제 3주차

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

height_weight = np.loadtxt('heights.csv',delimiter=',')
height_weight[:10,:]
x = height_weight[:10,0:1]
y = height_weight[:10,1:2]
line_fit = LinearRegression()
line_fit.fit(x,y)
pred_y=line_fit.predict(x)

print("기울기는",line_fit.coef_) #기울기
print("y절편은",line_fit.intercept_) #절편
plt.plot(x,y,'o')
plt.plot(x,pred_y)
plt.show()
