import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DISEASE_INFO = BASE_DIR / "disease_info" / "disease_info.json"


EMPTY_INFO = {
    "plant_name": "Không rõ",
    "disease_name": "Không rõ",
    "description": "Chưa có thông tin mô tả cho lớp dự đoán này.",
    "cause": "Chưa có thông tin nguyên nhân.",
    "solutions": "Chưa có thông tin giải pháp.",
    "prevention": "Chưa có thông tin phòng ngừa.",
}


def load_disease_info(info_path=DEFAULT_DISEASE_INFO):
    info_path = Path(info_path)
    if not info_path.exists():
        return {}

    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_disease_info(predicted_class, info_path=DEFAULT_DISEASE_INFO):
    data = load_disease_info(info_path)
    info = data.get(predicted_class, {})
    return {**EMPTY_INFO, **info}
