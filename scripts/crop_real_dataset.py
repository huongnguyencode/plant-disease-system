#!/usr/bin/env python3
"""Crop leaves from a class-folder dataset using a trained YOLO detector."""

import argparse
import sys
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_PADDING_RATIO = 0.08


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện và cắt vùng lá từ bộ ảnh thực tế bằng YOLO."
    )
    parser.add_argument("--input", default="real_data/raw", help="Thư mục ảnh đầu vào")
    parser.add_argument(
        "--output", default="real_data/cropped", help="Thư mục lưu ảnh đã cắt"
    )
    parser.add_argument(
        "--model",
        default="models/yolo_leaf_detector.pt",
        help="Đường dẫn model YOLO",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Ngưỡng tin cậy YOLO"
    )
    return parser.parse_args()


def iter_class_images(input_dir):
    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_dir.name, image_path


def padded_box(box, image_width, image_height, padding_ratio=DEFAULT_PADDING_RATIO):
    x1, y1, x2, y2 = box
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(image_width, int(x2 + pad_x + 0.5))
    y2 = min(image_height, int(y2 + pad_y + 0.5))
    return x1, y1, x2, y2


def crop_dataset(model, input_dir, output_dir, conf_threshold):
    total_images = 0
    total_crops = 0
    skipped_images = 0

    for class_name, image_path in iter_class_images(input_dir):
        total_images += 1
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_images += 1
            print(f"Bỏ qua ảnh không đọc được: {image_path}")
            continue

        try:
            results = model.predict(source=image, conf=conf_threshold, verbose=False)
        except Exception as exc:
            skipped_images += 1
            print(f"Lỗi YOLO, bỏ qua {image_path}: {exc}")
            continue

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            skipped_images += 1
            print(f"Không phát hiện lá, bỏ qua: {image_path}")
            continue

        height, width = image.shape[:2]
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        saved_for_image = 0

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue

            coordinates = box.xyxy[0].tolist()
            x1, y1, x2, y2 = padded_box(coordinates, width, height)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            output_path = output_class_dir / f"{image_path.stem}_crop{saved_for_image}.jpg"
            if not cv2.imwrite(
                str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            ):
                print(f"Không thể lưu ảnh crop: {output_path}")
                continue

            saved_for_image += 1
            total_crops += 1

        if saved_for_image == 0:
            skipped_images += 1
            print(f"Không có vùng lá hợp lệ, bỏ qua: {image_path}")

    return total_images, total_crops, skipped_images


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    model_path = Path(args.model)

    if not model_path.is_file():
        print(f"Lỗi: Không tìm thấy model YOLO tại {model_path}.", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Lỗi: Không tìm thấy thư mục ảnh đầu vào tại {input_dir}.", file=sys.stderr)
        return 1
    if not 0.0 <= args.conf <= 1.0:
        print("Lỗi: --conf phải nằm trong khoảng từ 0 đến 1.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Lỗi: Đã tìm thấy model YOLO nhưng chưa cài thư viện ultralytics.",
            file=sys.stderr,
        )
        return 1

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print(f"Lỗi: Không thể tải model YOLO: {exc}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    total_images, total_crops, skipped_images = crop_dataset(
        model=model,
        input_dir=input_dir,
        output_dir=output_dir,
        conf_threshold=args.conf,
    )

    print("\nTóm tắt:")
    print(f"Tổng số ảnh: {total_images}")
    print(f"Tổng số ảnh crop: {total_crops}")
    print(f"Số ảnh bị bỏ qua: {skipped_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
