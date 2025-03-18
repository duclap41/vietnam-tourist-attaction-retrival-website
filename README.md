# VIETNAM TOURIST DESTINATIONS WEB SEARCH

## Các bước cài đặt để chạy web demo:
### B1: Cài đặt các gói thư viện Python cần thiết
```pip install -r requirement.txt```
### B2: Mở terminal (powershell) ngay tại thư mục này, sau đó gõ lệnh để run server:
```uvicorn app:app```
### B3: Bấm vào đường link xuất hiện ngay bên dưới, thường có dạng:
```http://127.0.0.1:8000```
### B4: Bấm 'Ctrl + C' để stop server.
---
## Các bước để thực hiện truy vấn trên ứng dụng:
### B1: Tải lên một hoặc nhiều tấm ảnh về danh lam thắng cảnh Việt Nam (nằm trong 20 classes của đồ án), tất cả tấm ảnh trong một lượt truy vấn phải cùng liên quan đến một địa điểm.
### B2: Cải thiện chất lượng ảnh đã tải lên trước khi đi tìm searching, có chế độ Auto hoặc thủ công.
### B3: Chọn mô hình truy vấn và bắt đầu searching.
### B4: Website sẽ hiển thị một đoạn mô tả và top 10 ảnh liên quan đến địa điểm đó (có thể click vào để xem chi tiết ảnh).
### B5: Người dùng có thể re-ranking top 10 ảnh trả về bằng cách xóa những tấm ảnh không liên quan với ảnh tải lên hoặc đơn giản là thay đổi vị trí xếp hạng bằng cách kéo thả.
*Lưu ý: Phải xóa tất cả ảnh tải lên nếu muốn thực hiện một truy vấn khác!*

## Giải thích một số thư mục:
- Thư mục gốc sẽ chứa file backend và một số modules đơn giản.
- 'retrieval' chứa các module cung cấp hàm truy vấn.
- 'features' chứa các file giá trị đặc trưng đã được rút từ dataset.
- 'dataset' chứa tập train và test.
- 'weight' chứa trọng số của mô hình đã huấn luyện cho task phân lớp.
- 'baselines' chứa các file jupyter notebook dùng để EDA, huấn luyện cũng như đánh giá mô hình.
- 'templates' chứa file .html.
- 'static' chứa css, images, icons cho web.

Link Dataset:https://drive.google.com/drive/folders/1F7SLcJ41yI31NfDgLKMceYMYNPP7wr4Q?usp=drive_link
