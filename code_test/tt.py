import numpy as np
import os
import pickle
from tensorflow import keras
import matplotlib.pyplot as plt

# Dinh nghia bien
image_size = 256
#
data_path = 'data_test'
# Doc du lieu train, test tu file
with open("data.dat", "rb") as f:
    arr = pickle.load(f)
    noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]

# Load model autoencoder đã được train từ file:
autoencoder = keras.models.load_model("autoencoder.h5")


def load_normal_images(data_path):
    # trả về danh sách file trong thư mục đường dẫn
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path in normal_images_path:
        # Tạo đường dẫn đầy đủ đến tệp ảnh.
        full_img_path = os.path.join(data_path, img_path)
        # Sử dụng hàm image.load_img() của thư viện Keras để tải ảnh từ tệp
        # và thay đổi kích thước của ảnh thành img_size x img_size với chế độ màu grayscale.
        img = keras.preprocessing.image.load_img(full_img_path, target_size=(
            image_size, image_size), color_mode="grayscale")
        img = keras.preprocessing.image.img_to_array(
            img)  # Chuyển ảnh thành một mảng NumPy
        img = img / 255.0  # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        normal_images.append(img)  # Đưa vào list
    normal_images = np.array(normal_images)
    return normal_images


def make_noise(normal_image):
    mean = 0  # giá trị trung bình
    sigma = 1  # độ lệch chuẩn của phân phối Gaussian
    # tạo ra một ma trận các số ngẫu nhiên có phân phối chuẩn (Gaussian) với mean và sigma.
    gaussian = np.random.normal(mean, sigma, normal_image.shape)
    # định hình lại ma trận để có cùng hình dạng với hình ảnh đầu vào.
    noise_image = normal_image + gaussian*0.04
    return noise_image


def make_noise_images(normal_images):
    noise_images = []
    for img in normal_images:
        noise_image = make_noise(img)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images


def show_imageset(imageset):
    f, ax = plt.subplots(1, 1)
    ax.imshow(imageset[0].reshape(image_size, image_size), cmap="gray")
    plt.show()


# Load a noisy image
img_path = "./data_test/MQP_9963.JPG"
# img = image.load_img(img_path, target_size=(image_size, image_size))
# img = make_noise(noise_images)
normal_images = load_normal_images(data_path)
noise_images = make_noise_images(normal_images)
show_imageset(normal_images)
show_imageset(noise_images)
# Denoise the image
# denoised_img = autoencoder.predict(noise_images)

# # Display the original and denoised images

# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(noise_images[0].reshape(image_size, image_size), cmap="gray")

# plt.subplot(1, 2, 2)
# plt.title("Denoised Image")
# plt.imshow(denoised_img[0].reshape(image_size, image_size), cmap="gray")
# plt.show()
# Để chuyển ảnh sang không gian màu YUV và train bằng AutoEncoder để khử nhiễu ảnh và khôi phục lại màu sau khi khử nhiễu, ta cần cập nhật các hàm liên quan đến xử lý ảnh như sau:

# Hàm load_normal_images: thay đổi chế độ màu của ảnh sang YUV.
# Hàm generate_noise_data: thay đổi chế độ màu của ảnh sang YUV.
# Hàm denoise_image: thay đổi chế độ màu của ảnh đầu vào sang YUV, áp dụng model để khử nhiễu ảnh, sau đó chuyển đổi ảnh đã khử nhiễu về chế độ màu RGB.
# Hàm restore_color: chuyển đổi ảnh đầu vào sang chế độ màu YUV, khôi phục lại màu cho ảnh đã được khử nhiễu, sau
