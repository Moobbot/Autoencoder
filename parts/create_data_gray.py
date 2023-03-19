import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from tensorflow import keras


# Dinh nghia
data_path  = "data_testx512"
image_size  = 256 # Resize de tiet kiem thoi gian train
n_epochs = 100
n_batchsize = 32

# Load anh tu thu muc data
def load_normal_images(data_path):
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path  in normal_images_path:
        full_img_path = os.path.join(data_path, img_path)
        img = keras.preprocessing.image.load_img(full_img_path, target_size=(image_size, image_size), color_mode="grayscale")
        img = keras.preprocessing.image.img_to_array(img)
        img = img/255
        # Dua vao list
        normal_images.append(img)
    normal_images = np.array(normal_images)
    return normal_images

# Ham tao nhieu ngau nhien
def make_noise(normal_image, noise_factor):
    """
    Tạo ra ảnh nhiễu bằng cách thêm noise Gaussian vào  ảnh.
    Arguments:
        image: ảnh gốc
        noise_factor -- hệ số nhiễu
    Returns:
        image: ảnh góc được thêm nhiễu
    """

    # Tính độ lệch chuẩn của kênh Y
    sigma = np.std(normal_image)

    # Tạo nhiễu Gaussian với độ lệch chuẩn bằng noise_factor * sigma
    noise = np.random.normal(loc=0, scale=noise_factor*sigma, size=normal_image.shape)

    # Thêm nhiễu vào ảnh
    noisy_image = normal_image + noise
    
    # Cắt giá trị nằm ngoài khoảng [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image

# Ham tao tap du lieu noise
def make_noise_images(normal_images,noise_factor):
    noise_images = []
    for img in normal_images:
        noise_image = make_noise(img,noise_factor)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images

# How show thu du lieu
def show_imageset(imageset):
    f, ax = plt.subplots(1, 3)
    for i in range(1,4):
        ax[i-1].imshow(imageset[i].reshape(image_size,image_size), cmap="gray")
    plt.show()

data_link = 'data_test.dat'

if not os.path.exists(data_link):
    # Load anh normal
    normal_images = load_normal_images(data_path)
    # Tao anh noise
    noise_images = make_noise_images(normal_images, 0.3)

    # Chia du lieu train test
    noise_train, noise_test, normal_train, normal_test = train_test_split(noise_images, normal_images, test_size=0.2)
    with open(data_link, "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
else:
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
    plt.title('Normal image')
    plt.xticks([])
    plt.yticks([])
    plt.show()
