# Import thư viện hỗ trợ
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set parameters
# đường dẫn đến thư mục chứa dữ liệu huấn luyện
data_path = "data_train"
img_size = 128  # Resize để tiet kiem thoi gian train
n_epochs = 64  # Số lần train dữ liệu
# Batch size là số lượng các mẫu dữ liệu được đưa vào mạng để tính toán mỗi lần gradient descent.
n_batchsize = 32
model_name = 'autoencoder.h5'


# Create Function load image and prepare the dataset
def load_normal_images(data_path):
    # trả về danh sách file trong thư mục đường dẫn
    normal_images_path = os.listdir(data_path)
    normal_images = []
    for img_path in normal_images_path:
        # Tạo đường dẫn đầy đủ đến tệp ảnh.
        full_img_path = os.path.join(data_path, img_path)
        # Sử dụng hàm image.load_img() của thư viện Keras để tải ảnh từ tệp
        # và thay đổi kích thước của ảnh thành img_size x img_size với chế độ màu grayscale.
        img = image.load_img(full_img_path, target_size=(
            img_size, img_size), color_mode="grayscale")
        img = image.img_to_array(img)  # Chuyển ảnh thành một mảng NumPy
        img = img / 255.0  # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        normal_images.append(img)  # Đưa vào list
    normal_images = np.array(normal_images)
    return normal_images


# Create Function generate random noise for the image
def make_noise(normal_image):
    mean = 0  # giá trị trung bình
    sigma = 1  # độ lệch chuẩn của phân phối Gaussian
    # tạo ra một ma trận các số ngẫu nhiên có phân phối chuẩn (Gaussian) với mean và sigma.
    gaussian = np.random.normal(mean, sigma, normal_image.shape)
    # định hình lại ma trận để có cùng hình dạng với hình ảnh đầu vào.
    noise_image = normal_image + gaussian*0.04
    return noise_image


# Create Function data noise
def make_noise_images(normal_images):
    noise_images = []
    for img in normal_images:
        noise_image = make_noise(img)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images


# Create Function show image
def show_imageset(imageset):
    # Tạo đối tượng img "f", và 1 mảng 5 axes objects - 1 hàng images có 5 hình
    f, ax = plt.subplots(1, 5)
    for i in range(1, 6):
      # Hiển thị image và phương thức reshape() của image định hình lại mảng image 1D thành mảng 2D có kích thước (64, 64).
      # Đối số cmap="grey" chỉ định rằng colormap được sử dụng để hiển thị image là thang độ xám.
        ax[i-1].imshow(imageset[i].reshape(img_size, img_size), cmap="gray")
    plt.show()


# Define AutoEncoder model
def autoencoder():
    # khởi tạo một đối tượng Input với kích thước đầu vào là (image_size, image_size, 1)
    # Đối tượng này đại diện cho đầu vào của mô hình autoencoder và sẽ được sử dụng để xây dựng các tầng của mô hình
    inputs = Input(shape=(img_size, img_size, 1), name='image_input')
    # encoder- Gồm
    # 2 convolutional layers (Lớp tích chập)
    # và theo sau là max pooling layer (Lớp tổng hợp max)

    # convolutional layers 1 - Lấy img vầ sử dụng bộ lọc 64 kích thước 3x3 kích hoạt chức năng ReLU
    x = Conv2D(img_size, (3, 3), activation='relu',
               padding='same', name='Conv1')(inputs)
    # max pooling layer 1 - Nhận đầu ra của convolutional layers 1 chuyển qua max pooling layer kích thước 2x2
    x = MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    # convolutional layers 2 - Lấy đầu ra của max pooling layer 1 lặp lại thao tác
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='Conv2')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='pool2')(x)

  # decoder - Gồm
  # 2 convolutional layers (Lớp tích chập)
  # và theo sau là up-sampling layer (UpSampling2D) - Tăng kích thước đầu vào
  # Kích thước kernel 3x3, hàm kích hoạt relu và các kết nối đầu vào padding được giữ nguyên

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='Conv3')(encoded)
    # UpSampling2D kích thước 2x2 sử dụng để tăng kích thước input gấp đôi
    x = UpSampling2D((2, 2), name='upsample1')(x)
    x = Conv2D(img_size, (3, 3), activation='relu',
               padding='same', name='Conv4')(x)
    x = UpSampling2D((2, 2), name='upsample2')(x)
    # đưa ra đầu ra, có kích thước (image_size, image_size, 1) và hàm kích hoạt là sigmoid.
    decoded = Conv2D(1, (3, 3), activation='sigmoid',
                     padding='same', name='Conv5')(x)
    # model
    autoencoder = Model(inputs=inputs, outputs=decoded)
    # Biên dịch mô hình với hàm binary_crossentropy & tối ưu hóa bằng adam
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


# -------------------- Run -------------------- #

# Create modal AutoEncoder
denoise_model = autoencoder()
denoise_model.summary()

# If data has already been created and save to "data.dat", load the data
# Otherwise, load the normal training images and create noisy training data
# then save the data to "data.dat"

if os.path.exists('data.dat'):
    with open('data.dat', 'rb') as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]
else:
    # Load normal image
    normal_images = load_normal_images(data_path)

    # Create noise image
    noise_images = make_noise_images(normal_images)

    # Split the data into training and validation sets
    noise_train, noise_test, normal_train, normal_test = train_test_split(
        noise_images, normal_images, test_size=0.2)
    with open("data.dat", "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)

# Define EarlyStopping callback to stop training when validation loss stops improving
early_callback = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
# Model training
denoise_model.fit(noise_train, normal_train, epochs=n_epochs, batch_size=n_batchsize,
                  validation_data=(noise_test, normal_test),
                  callbacks=[early_callback])
denoise_model.save("denoise_model.h5")

# show_imageset(normal_images)
# show_imageset(noise_images)
# IUV - Đầu tiên chuyển màu của i sau đó được i' ghép với UV
