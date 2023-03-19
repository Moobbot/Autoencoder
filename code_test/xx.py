import numpy as np
import os
import pickle
from tensorflow import keras
from skimage import io, color
from skimage.transform import resize
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
        img = resize(img, (image_size, image_size, 3))
        normal_images.append(img)  # Thêm ảnh vào list
    normal_images = np.array(normal_images)  # Chuyển list thành numpy array
    return normal_images  # Trả về numpy array chứa dữ liệu ảnh YUV


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
