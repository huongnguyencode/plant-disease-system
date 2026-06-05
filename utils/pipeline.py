from pathlib import Path

from utils.classifier import predict_image
from utils.crop_leaf import crop_leaf_image
from utils.disease_mapping import get_disease_info
from utils.yolo_detect import detect_best_leaf_box


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CROP_DIR = BASE_DIR / "static" / "crops"


def _static_relative(path):
    path = Path(path).resolve()
    static_dir = (BASE_DIR / "static").resolve()
    return path.relative_to(static_dir).as_posix()


def run_pipeline(image_path, crop_dir=CROP_DIR):
    image_path = Path(image_path)

    try:
        bbox = detect_best_leaf_box(image_path)
        yolo_fallback = bbox is None
        crop_path = crop_leaf_image(image_path=image_path, crop_dir=crop_dir, bbox=bbox)
        prediction = predict_image(crop_path)
        disease_info = get_disease_info(prediction["predicted_class"])
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except RuntimeError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Đã xảy ra lỗi khi xử lý ảnh: {exc}"}

    return {
        "original_image": _static_relative(image_path),
        "crop_image": _static_relative(crop_path),
        "predicted_class": prediction["predicted_class"],
        "confidence": round(prediction["confidence"] * 100, 2),
        "yolo_fallback": yolo_fallback,
        **disease_info,
    }
