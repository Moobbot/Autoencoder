import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras

# Đường dẫn đến ảnh kiểm tra
# test_img_path = 'data_testx512/MQP_9562.jpg'
test_img_path = 'data_testx512/test_img2.jpg'

# Kich thước ảnh modal
image_size = 256

# Đọc ảnh kiểm tra
load_img = Image.open(test_img_path)
w,h = load_img.size

resized_image = load_img.resize((image_size, image_size))

# chuyển về không gian màu YCbCr
test_img = np.array(resized_image.convert('YCbCr')) / 255.

# Lấy kênh Y để làm ảnh gốc cho mô hình Autoencoder
test_img_y = test_img[:, :, 0]

# !Tạo ảnh nhiễu từ ảnh gốc

# Tính độ lệch chuẩn của kênh Y
sigma = np.std(test_img_y)

# noise_factor -- hệ số nhiễu
noise_factor = 0.2

# Tạo nhiễu Gaussian với độ lệch chuẩn bằng noise_factor * sigma
noise = np.random.normal(loc=0, scale=noise_factor*sigma, size=test_img_y.shape)

# test_img_y_noisy = test_img_y + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_img_y.shape)
test_img_y_noisy = test_img_y + noise
# test_img_y_noisy = test_img_y

# Cắt giá trị nằm ngoài khoảng [0, 255]
test_img_y_noisy = np.clip(test_img_y_noisy, 0, 255)

# Gộp kênh Y và các kênh Cb, Cr để tạo ra ảnh YCbCr sau khi được tạo nhiễu
noise_img = np.zeros_like(test_img)
noise_img[:, :, 0] = test_img_y_noisy
noise_img[:, :, 1:] = test_img[:, :, 1:]

# Reshape ảnh để đưa vào mô hình Autoencoder
test_img_y_noisy = np.expand_dims(test_img_y_noisy, axis=-1)
test_img_y_noisy = np.expand_dims(test_img_y_noisy, axis=0)

# Load model autoencoder đã được train từ file:
autoencoder = keras.models.load_model("denoise_model.h5")

# Khử nhiễu ảnh
denoised_img_y = autoencoder.predict(test_img_y_noisy)

# Reshape ảnh về dạng ban đầu
denoised_img_y = np.squeeze(denoised_img_y, axis=0)
denoised_img_y = np.squeeze(denoised_img_y, axis=-1)

# Gộp kênh Y và các kênh Cb, Cr để tạo ra ảnh YCbCr sau khi được khử nhiễu
denoised_img = np.zeros_like(test_img)
denoised_img[:, :, 0] = denoised_img_y
denoised_img[:, :, 1:] = test_img[:, :, 1:]

# Chuyển đổi ảnh YCbCr về RGB để hiển thị
noise_img = Image.fromarray((noise_img * 255.).astype(np.uint8), mode='YCbCr').convert('RGB')
denoised_img = Image.fromarray((denoised_img * 255.).astype(np.uint8), mode='YCbCr').convert('RGB')

# Khôi phục kích thước của ảnh
noise_img = noise_img.resize((w,h))
denoised_img = denoised_img.resize((w,h))

# Hiển thị ảnh gốc, ảnh nhiễu và ảnh sau khi được khử nhiễu

# plt.subplot(1, 3, 1)
# plt.imshow(load_img)
# plt.title('Original Image')

# plt.subplot(1, 3, 2)
# plt.imshow(noise_img)
# plt.title('Noise Image')

# plt.subplot(1, 3, 3)
# plt.imshow(denoised_img)
# plt.title('Denoised Image')

# plt.show()

plt.figure(figsize=(4,2))
plt.subplot(121)
plt.imshow(noise_img)
plt.title('Noise image')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(denoised_img)
plt.title('Denoised image')
plt.xticks([])
plt.yticks([])
plt.show()
