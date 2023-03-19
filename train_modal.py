# Import thư viện hỗ trợ
import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from tensorflow import keras


# Set parameters
# đường dẫn đến thư mục chứa dữ liệu huấn luyện
data_path  = "data_trainx512"
image_size  = 256  # Kích thước ảnh

# Create Function load image and prepare the dataset
def load_normal_images(data_path):
    # Trả về list name file trong thư mục data_path
    normal_images_path = os.listdir(data_path)
    normal_images = [] # list image
    # Duyệt qua từng tệp ảnh
    for img_path  in normal_images_path:
        # Tạo đường dẫn đầy đủ đến tệp ảnh
        full_img_path = os.path.join(data_path, img_path)
        # Đọc ảnh, Thay đổi kích thước của ảnh và chuyển ảnh sang kênh màu xám
        img = keras.preprocessing.image.load_img(
            full_img_path,
            target_size=(image_size, image_size),
            color_mode="grayscale")
        img = keras.preprocessing.image.img_to_array(img)  # Thêm ảnh vào list
        img = img/255 # Chuẩn hóa ảnh về khoảng giá trị [0, 1]
        normal_images.append(img) # Thêm ảnh vào list
    normal_images = np.array(normal_images) # Chuyển list thành numpy array
    return normal_images  # Trả về numpy array chứa dữ liệu ảnh

# Hàm tạo ra ảnh nhiễu
def make_noise(normal_image, noise_factor):
    """
    Tạo ra ảnh nhiễu bằng cách thêm noise Gaussian vào ảnh.
    Arguments:
        image: ảnh gốc
        noise_factor -- hệ số nhiễu
    Returns:
        image: ảnh góc được thêm nhiễu
    """
    # Tính độ lệch chuẩn của ảnh
    sigma = np.std(normal_image)
    # Tạo nhiễu Gaussian với độ lệch chuẩn bằng noise_factor * sigma
    noise = np.random.normal(loc=0, scale=noise_factor*sigma, size=normal_image.shape)
    # Thêm nhiễu vào ảnh
    noisy_image = normal_image + noise
    # Cắt giá trị nằm ngoài khoảng [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image

# Hàm tạo dữ liệu noise
def make_noise_images(normal_images,noise_factor):
    noise_images = []
    for img in normal_images:
        noise_image = make_noise(img,noise_factor)
        noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    return noise_images

# Hàm show thử dữ liệu
def show_imageset(imageset):
    f, ax = plt.subplots(1, 3)
    for i in range(1,4):
        ax[i-1].imshow(imageset[i].reshape(image_size,image_size), cmap="gray")
    plt.show()

data_link = 'data_gray.dat'
if not os.path.exists(data_link):
    print("Data not found. Creating new data...")
    # Load normal image
    normal_images = load_normal_images(data_path)
    # Create noise image
    noise_images = make_noise_images(normal_images, 0.3)

    # Split the data into 2 sets and test with a ratio of 80/20
    noise_train, noise_test, normal_train, normal_test = train_test_split(noise_images, normal_images, test_size=0.2)
    # Save data to data file.dat
    with open(data_link, "wb") as f:
        pickle.dump([noise_train, noise_test, normal_train, normal_test], f)
    print("Data created and saved successfully!")
else:
    print("Loading data...")
    # Load dữ liệu đã được tạo trước từ file
    with open(data_link, "rb") as f:
        arr = pickle.load(f)
        noise_train, noise_test, normal_train, normal_test = arr[0], arr[1], arr[2], arr[3]
    print("Data loaded successfully!")
    # show_imageset(noise_train)

    plt.figure(figsize=(8,2))
    plt.subplot(141)
    plt.imshow(noise_train[20], cmap='gray')
    plt.title('Noise image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(142)
    plt.imshow(normal_train[20], cmap='gray')
    plt.title('Normal image')
    plt.xticks([])
    plt.yticks([])

# ###################################### #
n_epochs = 100 # Kích thước ảnh
# Batch size là số lượng các mẫu dữ liệu được đưa vào mạng để tính toán mỗi lần gradient descent.
n_batchsize = 32

# Define AutoEncoder model
def create_autoencoder():
    input_img = keras.layers.Input(shape=(image_size, image_size, 1), name='image_input')

    # encoder: gồm hai tầng convolution và pooling để rút trích đặc trưng của ảnh.
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool1')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same', name='pool2')(x)

    # Tầng ẩn: gồm một tầng convolution để mã hóa đặc trưng của ảnh xuống một không gian giá trị thấp hơn.
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv3')(encoded)
    
    # Decoder: gồm hai tầng upsampling và một tầng convolution để tái tạo lại ảnh ban đầu từ đặc trưng đã được mã hóa.
    x = keras.layers.UpSampling2D((2, 2), name='upsample1')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv4')(x)
    x = keras.layers.UpSampling2D((2, 2), name='upsample2')(x)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='Conv5')(x)

    # Create modal AutoEncoder
    model = keras.models.Model(inputs=input_img, outputs=x)
    # Compile model với hàm binary_crossentropy & tối ưu hóa bằng adam
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    # Compile model với hàm binary_crossentropy & tối ưu hóa bằng mean_squared_error
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # Compile model với hàm binary_crossentropy & tối ưu hóa bằng adam
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

denoise_model = create_autoencoder()
denoise_model.summary()

# Train model
early_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta= 0 , patience=10, verbose=1, mode="auto")
denoise_model.fit(noise_train, normal_train,
                # Số lần lặp lại toàn bộ dữ liệu huấn luyện.
                epochs = n_epochs,
                # Kích thước của mỗi batch (tập con) được sử dụng trong quá trình huấn luyện.
                batch_size = n_batchsize,
                # Dữ liệu validation, được sử dụng để kiểm tra hiệu suất của mô hình trên dữ liệu mới.
                validation_data = (noise_test, normal_test),
                # Dừng huấn luyện sớm nếu không cải thiện được hiệu suất trên tập validation
                callbacks=[early_callback])

# Lưu mô hình
denoise_model.save("denoise_model.h5")