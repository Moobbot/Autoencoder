import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from tensorflow import keras


image_size  = 256

data_link = 'data_gray.dat'

with open(data_link, "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]
# show_imageset(noise_train)

plt.figure(figsize=(4,2))
plt.subplot(121)
plt.imshow(noise_train[20], cmap='gray')
plt.title('Noise image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(normal_train[20], cmap='gray')
plt.title('Original image')
plt.xticks([])
plt.yticks([])
plt.show()
