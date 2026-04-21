import json
import math
import os
import re
import uuid
from pathlib import Path

import flask
from PIL import Image, ImageFile, UnidentifiedImageError


DEFAULT_MAX_UPLOAD_MB = int(os.environ.get("TREEHEIGHT_MAX_UPLOAD_MB", "64"))
MAX_CONTENT_LENGTH = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.environ.get("TREEHEIGHT_MAX_IMAGE_PIXELS", "40000000"))
MAX_IMAGE_DIMENSION = int(os.environ.get("TREEHEIGHT_MAX_IMAGE_DIMENSION", "10000"))
MAX_NUMERIC_ABS = float(os.environ.get("TREEHEIGHT_MAX_NUMERIC_ABS", "1000000"))
TIMESTAMP_PATTERN = re.compile(r"^\d{10,20}$")
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
TEXT_EXTENSIONS = {".txt"}
VALID_IMAGE_FORMATS = {"JPEG", "MPO", "PNG"}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ValidationError(ValueError):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def configure_app_security(app):
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["MAX_FORM_MEMORY_SIZE"] = MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = os.environ.get("TREEHEIGHT_SECRET_KEY") or os.urandom(32)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("TREEHEIGHT_SECURE_COOKIE", "").lower() in {"1", "true", "yes"}


def add_security_headers(response):
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("Pragma", "no-cache")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def json_response(payload, status=200):
    return flask.Response(
        json.dumps(payload, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


def atomic_write_text(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_path(destination)
    with open(tmp_path, "w", encoding="utf-8") as tmp:
        tmp.write(content)
    os.replace(tmp_path, destination)


def cleanup_files(paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def validate_timestamp(raw_timestamp):
    timestamp = (raw_timestamp or "").strip()
    if not TIMESTAMP_PATTERN.fullmatch(timestamp):
        raise ValidationError("无效的请求标识")
    return timestamp


def parse_finite_float(raw_value, field_name, min_value=None, max_value=None):
    try:
        value = float((raw_value or "").strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} 不是有效数字")

    if not math.isfinite(value):
        raise ValidationError(f"{field_name} 不是有限数值")
    if abs(value) > MAX_NUMERIC_ABS:
        raise ValidationError(f"{field_name} 超出允许范围")
    if min_value is not None and value < min_value:
        raise ValidationError(f"{field_name} 小于允许范围")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} 大于允许范围")
    return value


def ordered_uploaded_files(request):
    return [storage for _, storage in request.files.items(multi=True)]


def image_timestamp_from_upload(file_storage):
    original_name = file_storage.filename or ""
    safe_name = Path(original_name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in JPEG_EXTENSIONS:
        raise ValidationError("仅支持 JPG/JPEG 图片上传")
    return validate_timestamp(Path(safe_name).stem)


def ensure_post_request(request):
    if request.method != "POST":
        raise ValidationError("请使用 POST 请求", status_code=405)


def save_verified_image(file_storage, target_path):
    try:
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as img:
            width, height = img.size
            img.verify()
        file_storage.stream.seek(0)
        with Image.open(file_storage.stream) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValidationError("图片尺寸无效")
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValidationError("图片尺寸超出允许范围", status_code=413)
            if img.format not in VALID_IMAGE_FORMATS:
                raise ValidationError("图片格式必须为 JPEG 或 PNG")

            if img.format in {"JPEG", "MPO"}:
                file_storage.stream.seek(0)
                save_upload_stream(file_storage, target_path)
            else:
                destination = Path(target_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = _temporary_path(destination, suffix=".jpg")
                try:
                    img.convert("RGB").save(tmp_path, format="JPEG", quality=95)
                    os.replace(tmp_path, destination)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValidationError("图片文件无效或过大", status_code=413)


def save_binary_upload(file_storage, target_path, allowed_extensions=None):
    safe_name = Path(file_storage.filename or "").name
    suffix = Path(safe_name).suffix.lower()
    if allowed_extensions and suffix not in allowed_extensions:
        raise ValidationError("上传文件类型不受支持")
    save_upload_stream(file_storage, target_path)


def save_upload_stream(file_storage, target_path):
    destination = Path(target_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temporary_path(destination)
    try:
        file_storage.stream.seek(0)
        file_storage.save(tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_point_cloud_path(output_path, model_type, timestamp):
    timestamp = validate_timestamp(timestamp)
    return os.path.join(output_path, f"{model_type}_{timestamp}_Depth_tree.ply")


def _temporary_path(destination, suffix=""):
    file_suffix = suffix or ".tmp"
    return destination.parent / f"tmp_{uuid.uuid4().hex}{file_suffix}"
