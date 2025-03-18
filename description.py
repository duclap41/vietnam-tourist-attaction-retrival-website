import requests

map_description = {
    0: "Lăng Chủ tịch Hồ Chí Minh",
    1: "Quần thể danh thắng Tràng An",
    2: "Kinh thành Huế",
    3: "Nhà thờ chính tòa Đức Bà Sài Gòn",
    4: "Vịnh Hạ Long",
    5: "Dinh Độc Lập",
    6: "Thánh địa Mỹ Sơn",
    7: "Hồ Hoàn Kiếm",
    8: "Bưu điện Sài Gòn",
    9: "Tượng Chúa Kitô Vua (Vũng Tàu)",
    10: "Cầu Vàng",
    11: "Cột cờ Lũng Cú",
    12: "Động Phong Nha",
    13: "Phố cổ Hội An",
    14: "Thác Bản Giốc",
    15: "Nhà thờ Lớn Hà Nội",
    16: "Chợ Bến Thành",
    17: "Ga Đà Lạt",
    18: "Chợ nổi",
    19: "Gành Đá Đĩa"
}

def get_wikipedia_content(class_number, lang="vi"):
    """
    Lấy nội dung chính về một địa điểm từ Wikipedia.

    Args:
    - title (str): Tên của bài viết Wikipedia.
    - lang (str): Ngôn ngữ của Wikipedia (mặc định là tiếng Việt).

    Returns:
    - str: Nội dung chính hoặc thông báo lỗi.
    """

    title = map_description[class_number]
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "exsectionformat": "plain",
        "inprop": "url"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "extract" in page_data:
                content = '\n'.join(page_data["extract"].split('\n')[:3])  # Lấy 3 đoạn đầu tiên của nội dung chính
                wiki_link = f"https://{lang}.wikipedia.org/wiki?curid={page_id}"
                return content, wiki_link
            elif "missing" in page_data:
                return f"Bài viết '{title}' không tồn tại trong Wikipedia {lang}.", None

        return "Không tìm thấy nội dung chi tiết cho địa điểm này.", None

    except requests.exceptions.RequestException as e:
        return f"Lỗi khi kết nối đến Wikipedia: {str(e)}", None

# Ví dụ sử dụng
if __name__ == "__main__":
    class_number = 17
    lang = 'vi'
    content, link = get_wikipedia_content(class_number, lang)
    print(content)
    print(link)


