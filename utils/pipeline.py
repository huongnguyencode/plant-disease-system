from pathlib import Path

from utils.classifier import predict_image
from utils.crop_leaf import crop_leaf_image
from utils.disease_mapping import get_disease_info
from utils.image_enhancement import enhance_crop_for_classifier, enhance_for_detection
from utils.image_quality import assess_image_quality
from utils.yolo_detect import detect_leaf_box


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CROP_DIR = BASE_DIR / "static" / "crops"
ENHANCED_DIR = BASE_DIR / "static" / "enhanced"


def _static_relative(path):
    path = Path(path).resolve()
    static_dir = (BASE_DIR / "static").resolve()
    return path.relative_to(static_dir).as_posix()


def run_pipeline(image_path, crop_dir=CROP_DIR, enhanced_dir=ENHANCED_DIR):
    image_path = Path(image_path)

    try:
        quality = assess_image_quality(image_path)
        if not quality["ok"]:
            return {
                "success": False,
                "error": quality["message"],
                "original_image": _static_relative(image_path),
                "enhanced_image": None,
                "crop_image": None,
                "classifier_image": None,
                "message": quality["message"],
                "quality_message": quality["message"],
                "quality_metrics": quality["metrics"],
                "quality_issues": quality["issues"],
            }

        enhanced_path = enhance_for_detection(
            image_path, save_dir=enhanced_dir, quality=quality
        )

        detection = detect_leaf_box(enhanced_path)
        crop_path = crop_leaf_image(
            image_path=enhanced_path,
            crop_dir=crop_dir,
            bbox=detection["bbox"],
        )
        classifier_crop_path = enhance_crop_for_classifier(
            crop_path, save_dir=enhanced_dir
        )
        prediction = predict_image(classifier_crop_path)
        predicted_class = prediction.get("class_name") or prediction.get(
            "predicted_class"
        )
        if not predicted_class:
            raise RuntimeError("Không tìm thấy nhãn dự đoán từ classifier.")
        disease_info = get_disease_info(predicted_class)
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": str(exc),
            "message": str(exc),
            "original_image": _static_relative(image_path),
        }
    except RuntimeError as exc:
        return {
            "success": False,
            "error": str(exc),
            "message": str(exc),
            "original_image": _static_relative(image_path),
        }
    except Exception as exc:
        message = f"Đã xảy ra lỗi khi xử lý ảnh: {exc}"
        return {
            "success": False,
            "error": message,
            "message": message,
            "original_image": _static_relative(image_path),
        }

    return {
        "success": True,
        "original_image": _static_relative(image_path),
        "enhanced_image": _static_relative(enhanced_path),
        "crop_image": _static_relative(crop_path),
        "classifier_image": _static_relative(classifier_crop_path),
        "predicted_class": predicted_class,
        "confidence": round(prediction["confidence"] * 100, 2),
        "low_confidence": prediction["confidence"] < 0.80,
        "quality_message": quality["message"],
        "quality_metrics": quality["metrics"],
        "quality_issues": quality["issues"],
        "yolo_fallback": detection["fallback"],
        "yolo_message": detection["message"],
        **disease_info,
    }
