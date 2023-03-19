import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

# Dinh nghia bien
image_size = 256

data_link = 'data_gray.dat'

# Doc du lieu train, test tu file
with open(data_link, "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

# Load model
model = load_model("denoise_model.h5")

# Chon random 5 anh de khu nhieu
s_id = 25
e_id = 27

pred_images = model.predict(noise_test[s_id: e_id])

# Ve len man hinh de kiem tra
for i in range(s_id, e_id):
    new_image = cv2.blur(noise_test[i], (3,3))
    new_image_1 = cv2.blur(noise_test[i], (5, 5))
    plt.figure(figsize=(5,3))
    plt.subplot(131)
    plt.imshow(pred_images[i-s_id].reshape(image_size ,image_size ,1), cmap='gray')
    plt.title('Model')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(132)
    plt.imshow(noise_test[i], cmap='gray')
    plt.title('Noise image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(normal_test[i], cmap='gray')
    plt.title('Normal image')
    plt.xticks([])
    plt.yticks([])

    plt.show()
