import pytube
import os


class vid:
    def __init__(self, url):
        self.url = url

    def is_valid_url(self):
        try:
            # Sử dụng phương thức validate_url để kiểm tra tính hợp lệ của URL
            pytube.YouTube(self.url)
            return True
        except pytube.exceptions.RegexMatchError:
            return False

    def download_video(self, video):
        # Liệt kê các định dạng và chất lượng video có thể tải về
        while True:
            format_dict = {}
            os.system("cls")
            total = 0
            for i, stream in enumerate(video.streams.filter().order_by('resolution')):
                format_dict[i] = stream
                print(f"{i}: {stream.resolution} ({stream.mime_type})")
                total = i

            # Nhập số thứ tự của định dạng và chất lượng muốn tải về
            try:
                format_index = int(
                    input("Chọn số thứ tự của chất lượng định dạng muốn tải về: "))
            except:
                continue

            if format_index < 0 or format_index > total:
                continue
            stream = format_dict[format_index]
            break

        # Tải về video
        stream.download()
        print("Tải video thành công!")

    def download_audio(self, video):
        # Liệt kê các định dạng và chất lượng âm thanh có thể tải về
        while True:
            format_dict = {}
            os.system("cls")
            total = 0
            for i, stream in enumerate(video.streams.filter(type="audio")):
                format_dict[i] = stream
                print(f"{i}: {stream.abr} ({stream.mime_type})")
                total = i

            # Nhập số thứ tự của định dạng và chất lượng muốn tải về
            try:
                format_index = int(
                    input("Chọn số thứ tự của chất lượng âm thanh muốn tải về: "))
            except:
                continue

            if format_index < 0 or format_index > total:
                continue
            stream = format_dict[format_index]
            break

        # Tải về audio
        stream.download()
        print("Tải audio thành công!")


def InvalidChoice():
    # các lựa chọn không xác định
    os.system("cls")
    os.system("color 4")
    print("==============================================================\n")
    print("                       Invalid Choice\n")
    print("==============================================================\n")
    os.system("pause")
    os.system("color 7")


def main():
    while True:
        os.system("cls")
        url = input("Nhập URL của video trên YouTube: ")
        os.system("cls")
        if not vid(url).is_valid_url():
            # kiểm tra xem URL có hợp lệ không, nếu không thì chạy lại
            print("URL không hợp lệ, vui lòng nhập lại.")
            os.system("pause")
            continue
        video = pytube.YouTube(url)
        while True:
            os.system("cls")
            print("==============================================================")
            print("[1] Audio")
            print("[2] Video")
            print("[3] Other URL")
            print("==============================================================")
            download_type = input("Your Choice: ")
            if download_type == "1":
                vid(url).download_audio(video=video)
            elif download_type == "2":
                vid(url).download_video(video=video)
            elif download_type == "3":
                break
            else:
                InvalidChoice()
                continue


if __name__ == "__main__":
    main()
