import pickle
import numpy as np
import os

from skimage import io, color
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# đường dẫn đến thư mục chứa dữ liệu huấn luyện
data_path = "data_train"

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
        img = resize(img, image_size)  # Thay đổi kích thước của ảnh
        img = img / 255.0  # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        img = color.rgb2yuv(img)  # Chuyển ảnh sang không gian màu YUV
        images.append(img)  # Thêm ảnh vào list
    images = np.array(images)  # Chuyển list thành numpy array
    return images


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


# Kiểm tra sự tồn tại của file data
if os.path.isfile('data.dat'):
    print("Loading data...")
    # Load dữ liệu đã được tạo trước từ file
    with open('data.dat', 'rb') as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]
    print("Data loaded successfully!")
else:
    print("Data not found. Creating new data...")
    # Load dữ liệu ảnh bình thường từ thư mục huấn luyện
    normal_train = load_data(data_path, image_size)
    noise_data = np.copy(normal_train)
    for i in range(noise_data.shape[0]):
        # Thay thế ảnh gốc bằng ảnh nhiễu
        noise_data[i] = add_noise(noise_data[i])
    # Chia dữ liệu train-test
    noise_train, noise_test, normal_train, normal_test = train_test_split(
        noise_data, normal_train, test_size=0.2)
    # Lưu dữ liệu vào file
    with open("data.dat", "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
    print("Data created and saved successfully!")
