from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_MODEL = BASE_DIR / "models" / "yolo_leaf_detector.pt"


def detect_best_leaf_box(image_path, model_path=DEFAULT_YOLO_MODEL, conf_threshold=0.25):
    """Return the best YOLO bounding box as (x1, y1, x2, y2), or None for fallback."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Chưa cài ultralytics. Vui lòng chạy: pip install -r requirements.txt") from exc

    model = YOLO(str(model_path))
    results = model(str(image_path), verbose=False)
    if not results:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

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
        return None

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    return int(x1), int(y1), int(x2), int(y2)
