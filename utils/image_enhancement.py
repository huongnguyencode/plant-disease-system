from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from utils.image_quality import build_leaf_mask


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENHANCED_DIR = BASE_DIR / "static" / "enhanced"


def _load_oriented_bgr(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _gamma_correct(image, gamma):
    table = np.array(
        [((value / 255.0) ** (1.0 / gamma)) * 255 for value in range(256)]
    ).astype(np.uint8)
    return cv2.LUT(image, table)


def _apply_mild_clahe(image, clip_limit=1.3):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=min(clip_limit, 1.5), tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )


def _compress_highlights(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_float = l_channel.astype(np.float32)
    bright = l_float > 205
    l_float[bright] = 205 + (l_float[bright] - 205) * 0.7
    return cv2.cvtColor(
        cv2.merge((np.clip(l_float, 0, 255).astype(np.uint8), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )


def _save_jpeg(image, output_path):
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Không thể lưu ảnh đã tiền xử lý: {output_path}")
    return output_path


def enhance_for_detection(image_path, save_dir=DEFAULT_ENHANCED_DIR, quality=None):
    """Create a mildly normalized image for leaf detection and cropping."""
    image_path = Path(image_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = _load_oriented_bgr(image_path)
    except Exception as exc:
        raise RuntimeError(f"Không thể đọc ảnh để tăng cường: {image_path}") from exc

    metrics = (quality or {}).get("metrics", {})
    issues = set((quality or {}).get("issues", []))
    mean_brightness = float(metrics.get("mean_brightness", 128.0))

    mask = build_leaf_mask(image)
    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)
    alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None] * 0.72

    # Keep color channels intact. Detection enhancement is limited to denoise and luminance.
    enhanced = cv2.bilateralFilter(image, d=5, sigmaColor=25, sigmaSpace=25)
    if "overexposed" in issues or mean_brightness > 190:
        enhanced = _compress_highlights(enhanced)
        enhanced = _gamma_correct(enhanced, gamma=0.97)
    elif "underexposed" in issues or mean_brightness < 70:
        enhanced = _gamma_correct(enhanced, gamma=1.08)

    enhanced = _apply_mild_clahe(enhanced, clip_limit=1.3)
    blended = (
        image.astype(np.float32) * (1.0 - alpha)
        + enhanced.astype(np.float32) * alpha
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    output_path = save_dir / f"enhanced_{image_path.stem}.jpg"
    return _save_jpeg(blended, output_path)


def enhance_crop_for_classifier(crop_path, save_dir=DEFAULT_ENHANCED_DIR):
    """Apply only a small luminance correction before EfficientNet inference."""
    crop_path = Path(crop_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        image = _load_oriented_bgr(crop_path)
    except Exception as exc:
        raise RuntimeError(f"Không thể đọc ảnh lá đã cắt: {crop_path}") from exc

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = build_leaf_mask(image)
    pixels = gray[mask > 0]
    mean_brightness = float(pixels.mean()) if pixels.size else float(gray.mean())

    corrected = image
    if mean_brightness < 65:
        corrected = _gamma_correct(image, gamma=1.06)
    elif mean_brightness > 200:
        corrected = _gamma_correct(image, gamma=0.96)

    output_path = save_dir / f"classifier_{crop_path.stem}.jpg"
    return _save_jpeg(corrected, output_path)


def enhance_image(image_path, save_dir=DEFAULT_ENHANCED_DIR, quality=None):
    """Backward-compatible alias for the detection enhancement stage."""
    return enhance_for_detection(image_path, save_dir=save_dir, quality=quality)
