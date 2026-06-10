import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parents[1]
REAL_WORLD_CLASSIFIER_MODEL = (
    BASE_DIR / "models" / "best_effnetv2s_real_finetuned.pth"
)
DEFAULT_CLASSIFIER_MODEL = BASE_DIR / "models" / "best_effnetv2s.pth"
DEFAULT_CLASS_NAMES = BASE_DIR / "models" / "class_names.json"


def load_class_names(class_names_path=DEFAULT_CLASS_NAMES):
    class_names_path = Path(class_names_path)
    if not class_names_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file tên lớp tại models/class_names.json. "
            "Vui lòng đặt file class_names.json vào đúng thư mục."
        )

    with class_names_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data[str(index)] for index in range(len(data))]

    raise ValueError("class_names.json phải là list hoặc object có key là chỉ số lớp.")


def build_model(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def resolve_classifier_model(model_path=None):
    if model_path is not None:
        return Path(model_path)
    if REAL_WORLD_CLASSIFIER_MODEL.exists():
        return REAL_WORLD_CLASSIFIER_MODEL
    return DEFAULT_CLASSIFIER_MODEL


def load_classifier(model_path=None, class_names_path=DEFAULT_CLASS_NAMES):
    model_path = resolve_classifier_model(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy model phân loại tại "
            "models/best_effnetv2s_real_finetuned.pth hoặc "
            "models/best_effnetv2s.pth."
        )

    print(f"Đang tải model phân loại: {model_path}")
    class_names = load_class_names(class_names_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names))

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        model = checkpoint
        state_dict = None

    if state_dict is not None:
        cleaned_state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model, class_names, device


def predict_image(image_path, model_path=None, class_names_path=DEFAULT_CLASS_NAMES):
    model, class_names, device = load_classifier(model_path, class_names_path)

    preprocess = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    class_index = int(class_index.item())
    return {
        "class_name": class_names[class_index],
        "predicted_class": class_names[class_index],
        "confidence": float(confidence.item()),
    }
