# Plant Disease Detection System

Hệ thống nhận diện bệnh trên lá cây sử dụng deep learning.

## Flow hệ thống

```text
User Upload Image
        ↓
YOLO Detect Leaf
        ↓
Crop Leaf
        ↓
EfficientNetV2-S Classifier
        ↓
Predicted Class
        ↓
Disease Info Mapping
        ↓
Web Result Page
Chức năng
Upload ảnh lá cây
Dự đoán bệnh trên lá
Hiển thị tên cây, tên bệnh, độ tin cậy
Hiển thị mô tả bệnh, nguyên nhân, giải pháp và phòng tránh
Có thể tích hợp YOLO để crop vùng lá
Cấu trúc project
plant-disease-system/
├── app.py
├── requirements.txt
├── models/
│   ├── best_effnetv2s.pth
│   ├── class_names.json
│   └── yolo_leaf_detector.pt
├── disease_info/
│   └── disease_info.json
├── utils/
├── templates/
└── static/
Cài đặt
git clone <YOUR_REPO_URL>
cd plant-disease-system

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
Thêm model

Tải model đã train và đặt vào thư mục:

models/
├── best_effnetv2s.pth
├── class_names.json
└── yolo_leaf_detector.pt

Nếu chưa có YOLO model, hệ thống có thể chạy bản classifier trước với:

best_effnetv2s.pth
class_names.json
Chạy app
python app.py

Mở trình duyệt:

http://127.0.0.1:5000
Dataset

Classifier được train bằng New Plant Diseases Dataset.

YOLO detector có thể train bằng PlantDoc Object Detection hoặc dataset bounding box tự gán nhãn.

Model
EfficientNetV2-S: phân loại bệnh lá cây
YOLOv8/YOLOv11: phát hiện và crop vùng lá

Lưu lại.

---

# 4. Kiểm tra `requirements.txt`

Mở file:

```bash
cat requirements.txt

Nếu chưa có hoặc thiếu, tạo lại:

nano requirements.txt

Dán:

flask
torch
torchvision
opencv-python
pillow
numpy
ultralytics
