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

        # Mở ảnh và thay đổi kích thước
        with Image.open(filepath) as img:
            img = img.resize((target_size, target_size))

            # Lưu ảnh với tên mới trong thư mục đầu ra
            output_filepath = os.path.join(output_dir, filename)
            img.save(output_filepath)


target_size = 512
image_dir = 'data_test'
output_dir = 'data_test'
resize_images(image_dir, output_dir, target_size)
print("Done resizing images to {}x{} in {}".format(
    target_size, target_size, output_dir))

# import cv2
# import os

# data_path = "data_train2"  # Thay đổi đường dẫn tới thư mục data

# image_count = 0
# for filename in os.listdir(data_path):
#     image_count += 1
# print("Total images in data folder: ", image_count)
