import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
CROP_DIR = BASE_DIR / "static" / "crops"
ENHANCED_DIR = BASE_DIR / "static" / "enhanced"
MODEL_DIR = BASE_DIR / "models"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR


def ensure_project_dirs():
    for directory in (UPLOAD_DIR, CROP_DIR, ENHANCED_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


ensure_project_dirs()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    image_file = request.files.get("image")
    if not image_file or image_file.filename == "":
        return render_template("index.html", error="Vui lòng chọn một ảnh lá cây.")

    if not allowed_file(image_file.filename):
        return render_template(
            "index.html",
            error="Định dạng ảnh không hợp lệ. Vui lòng dùng PNG, JPG, JPEG hoặc WEBP.",
        )

    ensure_project_dirs()

    original_name = secure_filename(image_file.filename)
    suffix = Path(original_name).suffix.lower()
    saved_name = f"{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / saved_name
    image_file.save(image_path)

    result = run_pipeline(image_path=image_path, crop_dir=CROP_DIR, enhanced_dir=ENHANCED_DIR)
    if result.get("error"):
        return render_template("result.html", result=result)

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
