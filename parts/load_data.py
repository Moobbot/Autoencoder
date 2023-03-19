import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# đường dẫn đến thư mục chứa dữ liệu huấn luyện
data_path = "data_trainx512"

# kích thước ảnh
image_size = (256, 256)

# số epoch
num_epochs = 100

# batch size
batch_size = 32


def load_data(data_path, image_size):
    """
    Load ảnh từ thư mục huấn luyện và chuyển sang không gian màu YUV
    Args:
        data_path: đường dẫn đến thư mục chứa ảnh huấn luyện
        image_size: kích thước của ảnh
    Returns:
        images: một numpy array chứa các ảnh đã chuyển sang không gian màu YUV
    """
    images = []
    for filename in os.listdir(data_path):
        img = io.imread(os.path.join(data_path, filename))  # Đọc ảnh
        # img = resize(img, image_size)# Thay đổi kích thước của ảnh
        # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        img = img.astype('float32') / 255.0
        img = color.rgb2yuv(img)  # Chuyển ảnh sang không gian màu YUV
        images.append(img)  # Thêm ảnh vào list
    images = np.array(images, dtype=object)  # Chuyển list thành numpy array
    # images = np.array(images)# Chuyển list thành numpy array
    return images


normal_train = load_data(data_path, image_size)
img = color.yuv2rgb(normal_train[5])
plt.subplot(1, 2, 1)
plt.imshow(np.clip(img, 0, 1))
plt.title('Original Image')


def load_data_link(link):
    with open(link, 'rb') as f:
        x_train, y_train, x_test, y_test = pickle.load(f)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data_link('data.dat')

noise_image = color.yuv2rgb(y_train[0])
pred_image = color.yuv2rgb(y_test[0])
plt.subplot(1, 4, 1)
plt.imshow(np.clip(color.yuv2rgb(x_train[0]), 0, 1))
plt.title('Noise Image')
plt.subplot(1, 4, 2)
plt.imshow(np.clip(color.yuv2rgb(y_train[0]), 0, 1))
plt.title('Original Image')
plt.subplot(1, 4, 3)
plt.imshow(np.clip(color.yuv2rgb(x_test[0]), 0, 1))
plt.title('Original Image')
plt.subplot(1, 4, 4)
plt.imshow(np.clip(color.yuv2rgb(y_test[0]), 0, 1))
plt.title('Original Image')
plt.show()
