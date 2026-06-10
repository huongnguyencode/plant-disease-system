# Project: Plant Disease Detection System

## Goal

Build a complete Flask web demo for plant leaf disease detection.

The final system must support this flow:

User Upload Image
-> Image Quality Analysis on Leaf Region
-> Leaf-Aware Image Restoration / Enhancement
-> YOLO Leaf Detection if model exists
-> Crop Leaf
-> EfficientNetV2-S Disease Classifier
-> Disease Info Mapping
-> Web Result Page

## Final Required Flow

The preprocessing flow must be:

Original Image
-> Fix EXIF Orientation
-> Build rough leaf/foreground mask using HSV saturation and brightness
-> Measure image quality on leaf/foreground region, not on white background
-> Reduce glare / compress highlights on leaf region
-> Apply moderate CLAHE
-> Slightly improve green color / contrast on leaf region
-> Apply mild sharpening
-> Send enhanced image to YOLO/fallback
-> Crop leaf
-> Send cropped leaf to EfficientNetV2-S classifier

## Important Behavior

1. The app must not reject readable images only because they are:

   * overexposed
   * underexposed
   * blurry
   * low contrast
   * tilted or rotated
   * affected by uneven lighting

2. The app must always try to restore/enhance readable images before YOLO and classification.

3. Only stop with an error if:

   * uploaded file cannot be read as an image
   * classifier model files are missing
   * classifier prediction fails

4. Image quality metrics must be calculated mainly on the leaf/foreground region, not on the whole background.

5. Enhancement should focus on the leaf region and avoid over-processing the white or dark background.

6. The enhanced image must be used as input for YOLO detection.

7. The cropped enhanced leaf image must be used as input for EfficientNetV2-S.

8. YOLO is optional.

   * If `models/yolo_leaf_detector.pt` exists, use YOLO.
   * If missing, fallback to full enhanced image box.
   * If YOLO fails to detect a box, fallback to full enhanced image box.
   * If `ultralytics` is missing, fallback without crashing.

9. EfficientNetV2-S must always be loaded with `weights=None`.

10. Do not download pretrained weights at runtime.

11. Do not require internet at runtime.

12. The web UI must be Vietnamese.

13. Do not commit `.pth` or `.pt` model files to git.

## Required Model Files

Classifier files are required:

* `models/best_effnetv2s.pth`
* `models/class_names.json`

YOLO file is optional:

* `models/yolo_leaf_detector.pt`

## Image Quality Module

File:

```text
utils/image_quality.py
```

Required functions:

```python
build_leaf_mask(image)
assess_image_quality(image_path)
```

### `build_leaf_mask(image)`

Build a rough leaf/foreground mask before YOLO.

Use OpenCV.

Recommended logic:

1. Convert image to HSV.
2. Convert image to grayscale.
3. Use saturation and brightness:

   * foreground if `saturation > 25`
   * or `gray < 245`
4. Use morphology open and close to remove noise.
5. Fill or smooth mask if needed.
6. If mask is too small, fallback to full image mask.

Return:

```python
mask
```

where mask is a uint8 image with values 0 or 255.

### `assess_image_quality(image_path)`

It must:

1. Read image using `cv2.imread`.
2. If unreadable, return `ok=False`.
3. Build leaf mask using `build_leaf_mask`.
4. Calculate metrics only on masked foreground pixels.
5. Calculate:

   * `bright_ratio`: ratio of masked grayscale pixels >= 245
   * `dark_ratio`: ratio of masked grayscale pixels <= 40
   * `blur_score`: Laplacian variance on masked/foreground region
   * `contrast_score`: grayscale std on masked/foreground region
   * `mean_brightness`: grayscale mean on masked/foreground region
   * `foreground_ratio`: mask area / full image area
6. Detect issues:

   * `overexposed` if `bright_ratio >= 0.25` or `mean_brightness > 190`
   * `underexposed` if `dark_ratio >= 0.30` or `mean_brightness < 70`
   * `blurry` if `blur_score < 80`
   * `low_contrast` if `contrast_score < 40`
   * `small_leaf` if `foreground_ratio < 0.08`

Return format:

```python
{
    "ok": bool,
    "can_enhance": bool,
    "message": str,
    "issues": list[str],
    "metrics": {
        "bright_ratio": float,
        "dark_ratio": float,
        "blur_score": float,
        "contrast_score": float,
        "mean_brightness": float,
        "foreground_ratio": float
    }
}
```

Readable images must return:

```python
"ok": True
"can_enhance": True
```

Only unreadable images should return:

```python
"ok": False
```

Vietnamese messages:

If image has quality issues:

```text
Ảnh có vấn đề về chất lượng, hệ thống đã tiền xử lý vùng lá trước khi phân tích.
```

If image has no quality issue:

```text
Ảnh đạt chất lượng tốt, hệ thống vẫn chuẩn hóa ảnh trước khi phân tích.
```

If image cannot be read:

```text
Không thể đọc ảnh đầu vào.
```

## Image Enhancement Module

File:

```text
utils/image_enhancement.py
```

Required functions:

```python
enhance_image(image_path, save_dir="static/enhanced", quality=None)
```

It must implement leaf-aware enhancement.

Enhancement steps:

1. Fix EXIF orientation using PIL `ImageOps.exif_transpose`.
2. Convert PIL RGB image to OpenCV BGR.
3. Build rough leaf mask using HSV saturation and brightness.
4. Create soft mask by Gaussian blur.
5. Apply light denoise using `cv2.bilateralFilter`.
6. Apply highlight compression on leaf region:

   * reduce very bright pixels on the leaf
   * compress LAB L or HSV V channel
   * avoid making the leaf too dark
7. Apply gamma correction:

   * brighten underexposed/dark images
   * reduce brightness slightly for overexposed/bright images
8. Apply moderate CLAHE on LAB L channel:

   * clipLimit between 1.2 and 1.8
   * avoid excessive contrast
9. Increase saturation slightly on leaf region if image is washed out.
10. Apply mild unsharp mask.
11. Apply stronger but still safe sharpening only if image is blurry.
12. Blend enhanced image back into original image using the soft mask:

* leaf region receives enhancement
* background remains close to original

13. Save to:

```text
static/enhanced/enhanced_<original_name>.jpg
```

14. Return enhanced image path.

Important:

* Do not over-process disease spots.
* Do not make the leaf unnaturally green.
* Do not sharpen too much.
* Do not process white background as if it were leaf.
* Overexposed white areas that have lost all detail cannot be fully recovered, but the system should still reduce glare and improve usable regions.

## YOLO Detection Module

File:

```text
utils/yolo_detect.py
```

Required function:

```python
detect_leaf_box(image_path, model_path="models/yolo_leaf_detector.pt", conf_threshold=0.25)
```

Return format:

```python
{
    "bbox": (x1, y1, x2, y2),
    "fallback": bool,
    "message": str
}
```

Behavior:

If YOLO model is missing:

```text
Không tìm thấy model YOLO. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.
```

If YOLO detects leaf:

```text
Đã phát hiện vùng lá bằng YOLO.
```

If YOLO detects no box:

```text
YOLO không phát hiện được lá. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.
```

If YOLO model exists but ultralytics is not installed:

```text
Có model YOLO nhưng chưa cài ultralytics. Hệ thống đã dùng toàn bộ ảnh làm vùng lá.
```

Rules:

* Do not import `ultralytics` if YOLO model file is missing.
* Do not crash if `ultralytics` is missing.
* Keep fallback mode stable.
* Keep `detect_best_leaf_box()` for backward compatibility if needed.

## Crop Module

File:

```text
utils/crop_leaf.py
```

The crop function must:

1. Receive enhanced image path.
2. Receive YOLO/fallback bounding box.
3. Clamp bounding box to image boundaries.
4. Crop the selected region.
5. Save crop to:

```text
static/crops/
```

6. Return crop path.

## Classifier Module

File:

```text
utils/classifier.py
```

Classifier requirements:

1. Use EfficientNetV2-S.
2. Always load model with:

```python
weights=None
```

3. Never use:

```python
EfficientNet_V2_S_Weights.IMAGENET1K_V1
```

4. Load checkpoint from:

```text
models/best_effnetv2s.pth
```

5. Load class names from:

```text
models/class_names.json
```

6. Return prediction with both keys:

```python
{
    "class_name": predicted_class,
    "predicted_class": predicted_class,
    "confidence": confidence
}
```

This prevents mismatch between classifier and pipeline.

## Pipeline Module

File:

```text
utils/pipeline.py
```

The pipeline must run in this exact order:

1. `quality = assess_image_quality(original_image)`
2. If `quality["ok"]` is false, return stable error result.
3. `enhanced_path = enhance_image(original_image, enhanced_dir, quality)`
4. `detection = detect_leaf_box(enhanced_path)`
5. `crop_path = crop_leaf_image(enhanced_path, crop_dir, detection["bbox"])`
6. `prediction = predict_image(crop_path)`
7. Read predicted class safely:

```python
predicted_class = prediction.get("class_name") or prediction.get("predicted_class")
```

8. If predicted class is missing, raise:

```python
RuntimeError("Không tìm thấy nhãn dự đoán từ classifier.")
```

9. `disease_info = get_disease_info(predicted_class)`
10. Return stable success result.

Success result must contain:

* `success`
* `original_image`
* `enhanced_image`
* `crop_image`
* `predicted_class`
* `confidence`
* `quality_message`
* `quality_metrics`
* `quality_issues`
* `yolo_fallback`
* `yolo_message`
* `plant_name`
* `disease_name`
* `description`
* `cause`
* `solutions`
* `prevention`

Error result must contain:

* `success: False`
* `error`
* `message`
* `original_image` if available

## Result Page

File:

```text
templates/result.html
```

When `result.success == True`, show:

* original image
* enhanced image
* cropped image
* quality message
* quality issues
* bright ratio
* dark ratio
* blur score
* contrast score
* mean brightness
* foreground ratio
* YOLO message
* YOLO fallback warning if fallback is true
* predicted class
* confidence
* plant name
* disease name
* description
* cause
* solutions
* prevention

Do not show a retake-photo page only because image is bright, dark, blurry, low contrast, or tilted.

When `result.success == False`, show the actual error message clearly.

## App

File:

```text
app.py
```

The app must ensure these folders exist:

* `static/uploads`
* `static/crops`
* `static/enhanced`
* `models`

The app must pass `enhanced_dir` to `run_pipeline`.

## Requirements

Required packages:

* flask
* torch
* torchvision
* opencv-python
* pillow
* numpy

Optional package:

* ultralytics

`ultralytics` must not be required unless YOLO model is present.

## Real-World Fine-tuning Plan

The current EfficientNetV2-S classifier was trained mainly on PlantVillage/New Plant Diseases style images. Real-world images can have complex backgrounds, multiple leaves, outdoor lighting, blur, occlusion, curled leaves, and small disease regions. Preprocessing alone cannot fully fix this domain shift.

The correct improvement plan is:

1. Use YOLO to detect and crop leaves from real-world images.
2. Save cropped leaves into class folders with the same names as the classes in `models/class_names.json`.
3. Fine-tune EfficientNetV2-S from the existing `models/best_effnetv2s.pth` checkpoint.
4. Use a small learning rate such as `1e-5` or `3e-5`.
5. Use realistic augmentation, including brightness, contrast, saturation, rotation, perspective, and blur.
6. Validate on a separate real-world validation set.

Do not over-process classifier input. The classifier-ready crop should preserve disease color and spot patterns. Model files must not be committed to git.

Expected folder structure:

```text
real_data/
  raw/
  cropped/
    train/
    valid/

notebooks/
  finetune_real_world_effnetv2s.ipynb

scripts/
  crop_real_dataset.py
```

## Validation

After code changes, always run:

```bash
python -m py_compile app.py utils/*.py
```

Then run:

```bash
python app.py
```
