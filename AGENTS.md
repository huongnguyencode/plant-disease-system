# Project: Plant Disease Detection System

## Goal
Build a complete plant leaf disease detection web demo.

The system has 4 main modules:

1. YOLOv8 or YOLOv11 leaf detector
- Detect leaf region from uploaded image.
- Select the best bounding box.
- Crop the leaf image.

2. EfficientNetV2-S classifier
- Classify cropped leaf image into disease classes.
- Load model from models/best_effnetv2s.pth.
- Load class names from models/class_names.json.

3. Disease Info Mapping
- Map predicted class to Vietnamese disease information.
- Load data from disease_info/disease_info.json.
- Return plant name, disease name, description, cause, solutions, and prevention.

4. Flask Web App
- User uploads image.
- Backend runs full pipeline.
- Web displays original image, cropped leaf image, predicted class, confidence, disease description, cause, solutions, and prevention.

## Tech stack
- Python
- Flask
- PyTorch
- Torchvision
- Ultralytics YOLO
- OpenCV
- Pillow
- HTML/CSS

## Project structure
plant-disease-system/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── yolo_leaf_detector.pt
│   ├── best_effnetv2s.pth
│   └── class_names.json
├── disease_info/
│   └── disease_info.json
├── utils/
│   ├── yolo_detect.py
│   ├── crop_leaf.py
│   ├── classifier.py
│   ├── disease_mapping.py
│   └── pipeline.py
├── templates/
│   ├── index.html
│   └── result.html
└── static/
    ├── uploads/
    ├── crops/
    └── style.css

## Rules
- Write clean and simple Python code.
- Handle errors clearly.
- If YOLO model is missing, allow fallback mode that uses the original image as crop image.
- If classifier model is missing, show a clear error message.
- Use Vietnamese text on the web UI.
- Keep the project runnable locally with `python app.py`.