# 🐔 Phát hiện bệnh gà qua phân bằng YOLO và Xception

## 📌 Giới thiệu

Dự án sử dụng **AI/Deep Learning** để phát hiện **các bệnh thường gặp ở gà thông qua ảnh phân**, kết hợp giữa hai mô hình:
- **YOLOv8**: Dùng để phát hiện vùng có thể chứa bệnh (phân gà).
- **Xception**: Phân loại loại bệnh từ ảnh vùng bệnh đã được cắt.

Ứng dụng được triển khai bằng **Streamlit**, cho phép người dùng:
- Tải lên ảnh hoặc chụp ảnh từ camera.
- Xử lý ảnh, phát hiện và phân loại bệnh.
- Xem thông tin chi tiết về bệnh, triệu chứng, cách điều trị và phòng ngừa.

---

## 🚀 Chức năng chính

✅ Tải ảnh hoặc chụp ảnh trực tiếp từ camera trên trình duyệt  
✅ Phát hiện vùng chứa phân gà bằng mô hình YOLO  
✅ Cắt và phân loại loại bệnh bằng mô hình Xception  
✅ Hiển thị vùng nghi ngờ mắc bệnh trên ảnh gốc  
✅ Tra cứu thông tin chi tiết về bệnh, triệu chứng, nguyên nhân, phòng ngừa và điều trị  
✅ Giao diện đơn giản, dễ dùng với Streamlit

---

## 🛠️ Công nghệ sử dụng

| Công nghệ | Vai trò |
|----------|--------|
| `YOLOv8` ([Ultralytics](https://github.com/ultralytics/ultralytics)) | Phát hiện vùng phân gà trên ảnh |
| `Xception` ([timm`](https://github.com/rwightman/pytorch-image-models)) | Mô hình phân loại bệnh từ ảnh |
| `Streamlit` | Xây dựng giao diện người dùng và triển khai ứng dụng |
| `Torch` / `Torchvision` | Xử lý tensor và tải mô hình |
| `Pillow` / `OpenCV` / `NumPy` | Tiền xử lý và hiển thị ảnh |
| `Python` | Ngôn ngữ chính để xây dựng ứng dụng |

---




