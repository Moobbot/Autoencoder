# Import thư viện hỗ trợ
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set parameters
# đường dẫn đến thư mục chứa dữ liệu huấn luyện
data_path = "data_train2"
img_size = (256, 256)  # Kích thước ảnh
n_epochs = 64  # Số lần train dữ liệu
# Batch size là số lượng các mẫu dữ liệu được đưa vào mạng để tính toán mỗi lần gradient descent.
n_batchsize = 32
model_name = 'autoencoder.h5'


# Create Function load image and prepare the dataset and convert image to YUV
def load_yuv_images(data_path):
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


# Hàm tạo ra ảnh nhiễu
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


# Define AutoEncoder model
def create_autoencoder(input_shape):
    input_img = keras.layers.Input(shape=input_shape, name='image_input')
    # encoder
    # 2 convolutional layers (Lớp tích chập) và theo sau là max pooling layer (Lớp tổng hợp max)
    # convolutional layers 1 - Lấy img vầ sử dụng bộ lọc 64 kích thước 3x3 kích hoạt chức năng ReLU
    x = keras.layers.Conv2D(32, (3, 3), activation='relu',
                            padding='same')(input_img)
    # max pooling layer 1 - Nhận đầu ra của convolutional layers 1 chuyển qua max pooling layer kích thước 2x2
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    # convolutional layers 2 - Lấy đầu ra của max pooling layer 1 lặp lại thao tác
    x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # decoder
    # 2 convolutional layers (Lớp tích chập) và theo sau là up-sampling layer (UpSampling2D) - Tăng kích thước đầu vào
    # Kích thước kernel 3x3, hàm kích hoạt relu và các kết nối đầu vào padding được giữ nguyên
    x = keras.layers.Conv2D(
        16, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(
        3, (3, 3), activation='sigmoid', padding='same')(x)
    # Create modal AutoEncoder
    autoencoder = keras.models.Model(input_img, decoded)
    # Compile model với hàm binary_crossentropy & tối ưu hóa bằng adadelta or adam
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


# Load normal image
# normal_images = load_yuv_images(data_path)

# Create noise image
# noise_images = add_noise(normal_images)

# plt.subplot(1, 2, 1)
# plt.imshow(normal_images[0][:, :, 0], cmap='gray')
# plt.show()
# -------------------- Run -------------------- #
# If data has already been created and save to "data.dat", load the data
# Otherwise, load the normal training images and create noisy training data
# then save the data to "data.dat"
if os.path.exists('data.dat'):
    with open('data.dat', 'rb') as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]
else:

    # Load normal image
    normal_images = load_yuv_images(data_path)

    # Create noise image
    noise_images = add_noise(normal_images)

    plt.subplot(1, 2, 1)
    plt.imshow(normal_images[0][:, :, 0], cmap='gray')
    # plt.imshow(color.yuv2rgb(normal_images[0]))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(noise_images[0][:, :, 0], cmap='gray')
    # plt.imshow(color.yuv2rgb(noise_images[0]))
    plt.title('Denoised Image')
    plt.show()
# Split the data into training and validation sets
noise_train, noise_test, normal_train, normal_test = train_test_split(
    noise_images, normal_images, test_size=0.2)
with open("data.dat", "wb") as f:
    pickle.dump([noise_train, noise_test, normal_train, normal_test], f)

# Create modal AutoEncoder
denoise_model = create_autoencoder()

denoise_model.fit(noise_train, normal_train,
                  # Số lần lặp lại toàn bộ dữ liệu huấn luyện.
                  epochs=n_epochs,
                  # Kích thước của mỗi batch (tập con) được sử dụng trong quá trình huấn luyện.
                  batch_size=n_batchsize,
                  # Xáo trộn dữ liệu huấn luyện sau mỗi lần lặp để tránh sự ảnh hưởng của thứ tự dữ liệu.
                  shuffle=True,
                  # Dữ liệu validation, được sử dụng để kiểm tra hiệu suất của mô hình trên dữ liệu mới.
                  validation_data=(noise_test, normal_test),
                  # Dừng huấn luyện sớm nếu không cải thiện được hiệu suất trên tập validation
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode="auto")])
# Lưu mô hình
denoise_model.save(model_name)
