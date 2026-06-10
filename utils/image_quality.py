from pathlib import Path

import cv2
import numpy as np


BRIGHT_PIXEL_THRESHOLD = 245
DARK_PIXEL_THRESHOLD = 40
SATURATION_FOREGROUND_THRESHOLD = 25
BACKGROUND_GRAY_THRESHOLD = 245
MIN_FOREGROUND_RATIO_FOR_MASK = 0.01
OVEREXPOSED_RATIO_THRESHOLD = 0.25
UNDEREXPOSED_RATIO_THRESHOLD = 0.30
BLUR_SCORE_THRESHOLD = 80.0
LOW_CONTRAST_THRESHOLD = 40.0
SMALL_LEAF_THRESHOLD = 0.08


def _raw_leaf_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    mask = np.where(
        (saturation > SATURATION_FOREGROUND_THRESHOLD)
        | (gray < BACKGROUND_GRAY_THRESHOLD),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_leaf_mask(image):
    mask = _raw_leaf_mask(image)
    foreground_ratio = float(np.count_nonzero(mask) / mask.size)
    if foreground_ratio < MIN_FOREGROUND_RATIO_FOR_MASK:
        return np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def _empty_quality_result(message):
    return {
        "ok": False,
        "can_enhance": False,
        "message": message,
        "issues": [],
        "metrics": {
            "bright_ratio": 0.0,
            "dark_ratio": 0.0,
            "blur_score": 0.0,
            "contrast_score": 0.0,
            "mean_brightness": 0.0,
            "foreground_ratio": 0.0,
        },
    }


def assess_image_quality(image_path):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return _empty_quality_result("Không thể đọc ảnh đầu vào.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_mask = _raw_leaf_mask(image)
    mask = build_leaf_mask(image)
    raw_foreground_ratio = float(np.count_nonzero(raw_mask) / raw_mask.size)
    foreground_ratio = raw_foreground_ratio
    foreground_pixels = gray[mask > 0]

    if foreground_pixels.size == 0:
        foreground_pixels = gray.reshape(-1)
        foreground_ratio = 1.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    foreground_laplacian = laplacian[mask > 0]
    if foreground_laplacian.size == 0:
        foreground_laplacian = laplacian.reshape(-1)

    bright_ratio = float(np.mean(foreground_pixels >= BRIGHT_PIXEL_THRESHOLD))
    dark_ratio = float(np.mean(foreground_pixels <= DARK_PIXEL_THRESHOLD))
    blur_score = float(foreground_laplacian.var())
    contrast_score = float(foreground_pixels.std())
    mean_brightness = float(foreground_pixels.mean())

    issues = []
    if bright_ratio >= OVEREXPOSED_RATIO_THRESHOLD or mean_brightness > 190:
        issues.append("overexposed")
    if dark_ratio >= UNDEREXPOSED_RATIO_THRESHOLD or mean_brightness < 70:
        issues.append("underexposed")
    if blur_score < BLUR_SCORE_THRESHOLD:
        issues.append("blurry")
    if contrast_score < LOW_CONTRAST_THRESHOLD:
        issues.append("low_contrast")
    if foreground_ratio < SMALL_LEAF_THRESHOLD:
        issues.append("small_leaf")

    metrics = {
        "bright_ratio": round(bright_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
        "blur_score": round(blur_score, 2),
        "contrast_score": round(contrast_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "foreground_ratio": round(foreground_ratio, 4),
    }

    message = (
        "Ảnh có vấn đề về chất lượng, hệ thống đã tiền xử lý vùng lá trước khi phân tích."
        if issues
        else "Ảnh đạt chất lượng tốt, hệ thống vẫn chuẩn hóa ảnh trước khi phân tích."
    )

    return {
        "ok": True,
        "can_enhance": True,
        "message": message,
        "issues": issues,
        "metrics": metrics,
    }
