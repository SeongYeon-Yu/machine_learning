#60181912 유성연
#기계학습응용 과제
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits= load_digits(n_class=6)

one_idx = np.argwhere(digits.target ==1)
fig, ax = plt.subplots(5,5, figsize=(6,6))
j=1

for i in range(int(one_idx.size)):
    if i in one_idx:
        plt.subplot (5,5, j)
        image = digits.images[i]
        plt.imshow(image[1:7,2:6], cmap ='binary')
        j += 1
    if j> 25:
        break
plt.show()
