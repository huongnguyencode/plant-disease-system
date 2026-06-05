from pathlib import Path

from PIL import Image


def crop_leaf_image(image_path, crop_dir, bbox=None):
    """Crop the detected leaf box. If bbox is None, save the original image as crop."""
    image_path = Path(image_path)
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(1, min(x2, width))
        y2 = max(1, min(y2, height))
        if x2 > x1 and y2 > y1:
            image = image.crop((x1, y1, x2, y2))

    crop_name = f"crop_{image_path.stem}.jpg"
    crop_path = crop_dir / crop_name
    image.save(crop_path, format="JPEG", quality=95)
    return crop_path
