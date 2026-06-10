from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL = BASE_DIR / "models" / "yolo_leaf_detector.pt"


def _full_image_box(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return 0, 0, width, height


def detect_leaf_box(image_path, model_path=DEFAULT_YOLO_MODEL, conf_threshold=0.25):
    """Return detection data. Use the full image when YOLO is unavailable or misses."""
    image_path = Path(image_path)
    model_path = Path(model_path)
    if not model_path.exists():
        return {
            "bbox": _full_image_box(image_path),
            "fallback": True,
            "message": "Không tìm thấy model YOLO. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.",
        }

    try:
        from ultralytics import YOLO
    except ImportError:
        return {
            "bbox": _full_image_box(image_path),
            "fallback": True,
            "message": "Có model YOLO nhưng chưa cài ultralytics. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.",
        }

    model = YOLO(str(model_path))
    results = model(str(image_path), verbose=False)
    if not results:
        return {
            "bbox": _full_image_box(image_path),
            "fallback": True,
            "message": "YOLO không phát hiện được lá. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.",
        }

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {
            "bbox": _full_image_box(image_path),
            "fallback": True,
            "message": "YOLO không phát hiện được lá. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.",
        }

    best_box = None
    best_conf = -1.0
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return {
            "bbox": _full_image_box(image_path),
            "fallback": True,
            "message": "YOLO không tìm thấy vùng lá đủ tin cậy. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.",
        }

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    return {
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
        "fallback": False,
        "message": "Đã phát hiện vùng lá bằng YOLO.",
    }


def detect_best_leaf_box(image_path, model_path=DEFAULT_YOLO_MODEL, conf_threshold=0.25):
    """Return the best leaf bounding box as (x1, y1, x2, y2)."""
    return detect_leaf_box(image_path, model_path, conf_threshold)["bbox"]
