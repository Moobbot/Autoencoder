import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# URL của trang web cần tải
url = 'https://example.com'

# Tải nội dung HTML của trang web
html = requests.get(url).content

# Phân tích cú pháp HTML của trang web
soup = BeautifulSoup(html, 'html.parser')

# Tìm tất cả các thẻ hình ảnh (img) trên trang web
img_tags = soup.find_all('img')

# Tạo một danh sách các URL ảnh
img_urls = [img['src'] for img in img_tags]

# Tải xuống tất cả các ảnh và lưu vào thư mục hiện tại
for img_url in img_urls:
    # Kiểm tra xem URL ảnh có phải là URL tương đối hay tuyệt đối
    if not bool(urlparse(img_url).netloc):
        # Nếu URL là tương đối, chuyển đổi nó thành URL tuyệt đối
        img_url = urljoin(url, img_url)
    # Tạo đường dẫn lưu trữ cho ảnh
    filename = img_url.split("/")[-1]
    filepath = "./" + filename
    # Tải ảnh xuống và lưu vào đường dẫn lưu trữ
    response = requests.get(img_url)
    with open(filepath, "wb") as f:
        f.write(response.content)
