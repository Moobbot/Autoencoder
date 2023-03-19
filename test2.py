# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Đường dẫn tới file ảnh
# img_path = "data_test/MQP_9963.JPG"

# # Đọc ảnh sử dụng PIL
# img = Image.open(img_path)

# # Chuyển đổi sang không gian màu YUV
# yuv_img = img.convert("YCbCr")

# # Chuyển đổi sang numpy array
# yuv_arr = np.asarray(yuv_img)

# # Lấy kênh Y (độ xám) và chuẩn hóa về đoạn [0, 1]
# y_channel = yuv_arr[:, :, 0]
# y_channel_norm = y_channel/255.0

# # Hiển thị ảnh độ xám
# plt.imshow(y_channel_norm, cmap='gray')

# # # Lấy kênh U và V và chuẩn hóa về đoạn [0, 1]
# # u_channel = yuv_arr[:, :, 1]
# # v_channel = yuv_arr[:, :, 2]
# # uv_channel_norm = np.stack((u_channel, v_channel), axis=-1)/255.0

# # # Hiển thị ảnh kênh U và V
# # plt.imshow(uv_channel_norm)
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage import io, color
import matplotlib.pyplot as plt
import os
import numpy as np
plt.show()


def add_noise(Y, noise_factor):
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


img = io.imread('data_test/MQP_9963.JPG')
img_nomal = img
img = img / 255.0
img = color.rgb2yuv(img_nomal)
plt.subplot(1, 2, 1)
plt.imshow(img_nomal)
plt.title('Original Image')
# img_yuv_y = img[:, :, 0]
# img[:, :, 0] = add_noise(img_yuv_y, 0.3)
# img = color.yuv2rgb(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img)
# plt.title('Noise Image')
# print(img)
plt.show()
