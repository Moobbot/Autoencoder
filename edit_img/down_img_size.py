from PIL import Image
import os

data_path = "data_train"  # Thay đổi đường dẫn tới thư mục data

for filename in os.listdir(data_path):
    full_img_path = os.path.join(data_path, filename)
    # Load ảnh
    img = Image.open(full_img_path)
    file_size = os.path.getsize(full_img_path)/1024/1024
    if (file_size > 5):
        # Lưu ảnh mới với định dạng JPEG và chất lượng nén ảnh 80%
        img.save(full_img_path, format="JPEG", quality=80)
