import os
from PIL import Image

# Đường dẫn đến thư mục chứa ảnh
image_folder_path = "data_train"

# Định dạng ảnh mới (ví dụ: .jpg)
new_format = ".jpg"

# Lặp qua tất cả các tệp trong thư mục
for filename in os.listdir(image_folder_path):
    # Đường dẫn đến tệp ảnh
    file_path = os.path.join(image_folder_path, filename)

    # Nếu là tệp ảnh
    if os.path.isfile(file_path) and filename.lower().endswith((".png", ".jpeg", ".gif", ".JPG")):
        # Mở ảnh
        with Image.open(file_path) as img:
            # Tạo đường dẫn mới với định dạng mới
            new_file_path = os.path.join(
                image_folder_path, f"{os.path.splitext(filename)[0]}{new_format}")
            # Lưu ảnh với định dạng mới
            img.save(new_file_path)
            # Xóa tệp ảnh cũ
            os.remove(file_path)
