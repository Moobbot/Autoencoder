import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Dinh nghia bien
image_size = 100

# Get data train, test from file
# noise_train: Mảng chứa các hình ảnh nhiễu dùng để train autoencoder.
# noise_test: Mảng chứa các hình ảnh nhiễu dùng để test autoencoder.
# normal_train: Mảng chứa các hình ảnh không nhiễu tương ứng với các hình ảnh trong noise_train.
# normal_test: Mảng chứa các hình ảnh không nhiễu tương ứng với các hình ảnh trong noise_test.
with open("data.dat", "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

# Load model autoencoder đã được train từ file:
autoencoder = load_model("denoise_model.h5")

# Chon random 5 anh de khu nhieu
s_id = 0
e_id = 4

pred_images = autoencoder.predict(noise_test[s_id: e_id])
# # Chuẩn hóa ảnh và chuyển đổi kích thước
# pred_images = normalize(img_to_array(pred_images))
# pred_images = cv2.resize(pred_images, (200, 200))
# # Ve len man hinh de kiem tra
for i in range(s_id, e_id):
    new_image = cv2.blur(noise_test[i], (3, 3))
    new_image_1 = cv2.blur(noise_test[i], (5, 5))
    plt.figure(figsize=(8, 3))
    plt.subplot(141)
    plt.imshow(pred_images[i-s_id].reshape(image_size,
               image_size), cmap='gray')
    plt.title('Model')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(142)
    plt.imshow(new_image, cmap='gray')
    plt.title('Blur OpenCV (K3)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(143)
    plt.imshow(new_image_1, cmap='gray')
    plt.title('Blur OpenCV (K5)')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(144)
    plt.imshow(noise_test[i], cmap='gray')
    plt.title('Noise image')
    plt.xticks([])
    plt.yticks([])

    plt.show()
