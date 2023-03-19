import numpy as np
import os
import pickle

from tensorflow import keras
import matplotlib.pyplot as plt

# Dinh nghia bien
image_size = 256
data_path = 'data_test'
# Doc du lieu train, test tu file
with open("data.dat", "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

# Load model autoencoder đã được train từ file:
autoencoder = keras.models.load_model("autoencoder.h5")


def load_normal_images(data_path):
    # Trả về list name file trong thư mục data_path
    normal_images_path = os.listdir(data_path)
    # list image
    normal_images = []
    # Duyệt qua từng tệp ảnh
    for img_path in normal_images_path:
        # Tạo đường dẫn đầy đủ đến tệp ảnh
        full_img_path = os.path.join(data_path, img_path)
        img = io.imread(full_img_path)  # Đọc ảnh
        img = img / 255.0  # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        img = color.rgb2yuv(img)  # Chuyển ảnh sang không gian màu YUV
        # Thay đổi kích thước của ảnh
        img = resize(img, (img_size[0], img_size[1], 3))
        normal_images.append(img)  # Thêm ảnh vào list
    normal_images = np.array(normal_images)  # Chuyển list thành numpy array
    return normal_images  # Trả về numpy array chứa dữ liệu ảnh YUV


def add_noise(images):
    # Tạo ImageDataGenerator để tạo ra ảnh nhiễu
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    # Lặp qua các ảnh trong dữ liệu huấn luyện
    for i in range(images.shape[0]):
        # Reshape ảnh để phù hợp với đầu vào của ImageDataGenerator
        img = images[i].reshape(1, img_size[0], img_size[1], 3)
        # Tạo ra ảnh nhiễu từ ảnh gốc
        noisy_img = datagen.flow(img, batch_size=1).next()[
            0].reshape(img_size[0], img_size[1], 3)
        # Thay thế ảnh gốc bằng ảnh nhiễu
        images[i] = noisy_img
    # Trả về dữ liệu ảnh nhiễu
    return images


def show_imageset(imageset):
    plt.subplot(1, 2, 1)
    plt.imshow(imageset[0][:, :, 0], cmap='gray')
    # plt.imshow(imageset[0])
    plt.show()


# Load a noisy image
# img_path = "./data_test/MQP_9958.JPG"
# normal_images = keras.preprocessing.image.load_img(img_path, target_size=(image_size, image_size))
normal_images = load_normal_images(data_path)
noise_images = add_noise(normal_images)
# show_imageset(normal_images)
# show_imageset(noise_images)
# Denoise the image
denoised_img = autoencoder.predict(noise_images)

# Display the original and denoised images

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(noise_images[0][:, :, 0], cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised_img[0][:, :, 0], cmap="gray")
plt.show()
