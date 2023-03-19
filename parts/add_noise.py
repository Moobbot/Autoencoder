import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


def add_noise(image, noise_factor):
    """
    Tạo ra ảnh nhiễu bằng cách thêm noise Gaussian vào kênh Y của ảnh.
    Arguments:
        image: ảnh gốc
        Y -- kênh Y của ảnh YUV (numpy array)
        noise_factor -- hệ số nhiễu
    Returns:
        image: ảnh góc được thêm nhiễu
    """
    # Lấy kênh Y của ảnh
    Y = image[:, :, 0]

    # Tính độ lệch chuẩn của kênh Y
    sigma = np.std(Y)

    # Tạo nhiễu Gaussian với độ lệch chuẩn bằng noise_factor * sigma
    noise = np.random.normal(loc=0, scale=noise_factor*sigma, size=Y.shape)

    # Thêm nhiễu vào kênh Y
    Y_noisy = Y + noise

    # Cắt giá trị nằm ngoài khoảng [0, 255]
    Y_noisy = np.clip(Y_noisy, 0, 255)

    # Ghi đè kênh Y của ảnh bằng kênh Y nhiễu
    image[:, :, 0] = Y_noisy

    return image


def add_Y_noise(Y, noise_factor):
    """
    Tạo ra ảnh nhiễu cho kênh Y bằng cách thêm nhiễu Gaussian.

    Arguments:
    Y -- kênh Y của ảnh YUV (numpy array)
    noise_factor -- hệ số nhiễu

    Returns:
    Y_noisy -- kênh Y đã được thêm nhiễu (numpy array)
    """
    # Tính độ lệch chuẩn của kênh Y
    sigma = np.std(Y)

    # Tạo nhiễu Gaussian với độ lệch chuẩn bằng noise_factor * sigma
    noise = np.random.normal(loc=0, scale=noise_factor*sigma, size=Y.shape)

    # Thêm nhiễu vào kênh Y
    Y_noisy = Y + noise

    # Cắt giá trị nằm ngoài khoảng [0, 255]
    Y_noisy = np.clip(Y_noisy, 0, 255)

    return Y_noisy


# EX
img = io.imread('data_test/MQP_9559.JPG')
img_nomal = img
img = img / 255.0
plt.subplot(1, 2, 1)
plt.imshow(img_nomal)
plt.title('Original Image')
img = color.rgb2yuv(img_nomal)
# img_yuv_y = img[:,:,0]
# img[:,:,0] = add_noise(img_yuv_y,0.5)
img = add_noise(img, 0.5)
img = color.yuv2rgb(img)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.title('Noise Image')
# print (img)
