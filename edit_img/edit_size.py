import cv2
from PIL import Image
import os


def resize_images(image_dir, output_dir, target_size):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lặp qua tất cả các tệp ảnh trong thư mục đầu vào
    for filename in os.listdir(image_dir):
        # Tạo đường dẫn đầy đủ đến tệp ảnh
        filepath = os.path.join(image_dir, filename)

        # Kiểm tra xem đó có phải là tệp ảnh hay không
        if not (filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".png") or filepath.endswith(".JPG")):
            continue

        # Load ảnh từ file
        img = cv2.imread(filepath)

        # Lấy kích thước ảnh
        height, width, channels = img.shape
        ratio = width/height
        if (width > target_size):
            width = target_size
            height = int(target_size/ratio)
        else:
            if (height > target_size):
                height = target_size
                width = int(height*ratio)
                # img = cv2.flip(img, 1)

        # Mở ảnh và thay đổi kích thước
        with Image.open(filepath) as img:
            img = img.resize((width, height))

            # Lưu ảnh với tên mới trong thư mục đầu ra
            output_filepath = os.path.join(output_dir, filename)
            img.save(output_filepath)


target_size = 512
image_dir = 'data_train'
output_dir = 'data_trainx512'
resize_images(image_dir, output_dir, target_size)
print("Done resizing images to {}x{} in {}".format(
    target_size, target_size, output_dir))
