# -*- coding: UTF-8 -*-
import json
import os
import threading

import flask
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import calculate as calc
import image_util
from web_security import (
    DEFAULT_MAX_UPLOAD_MB,
    TEXT_EXTENSIONS,
    ValidationError,
    add_security_headers,
    atomic_write_text,
    build_point_cloud_path,
    cleanup_files,
    configure_app_security,
    ensure_post_request,
    image_timestamp_from_upload,
    json_response,
    ordered_uploaded_files,
    parse_finite_float,
    save_binary_upload,
    save_verified_image,
)


app = flask.Flask(__name__)
configure_app_security(app)

if not torch.cuda.is_available():
    print("警告: CUDA 不可用，将使用 CPU")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")

H = 90
W = 160
scale = 90 / 1080

uploads_path = "uploads"
input_path = os.path.join(uploads_path, "input")
output_path = os.path.join(uploads_path, "output")

model_type = "MobileDepth"
_startup_done = False
_startup_lock = threading.Lock()

LEGACY_DATA_SUFFIXES = {
    1: "_depthdata.txt",
    2: "_rawdepthdata.txt",
    3: "_confidencedata.txt",
}


@app.before_request
def startup():
    global _startup_done

    if _startup_done:
        return

    with _startup_lock:
        if _startup_done:
            return

        if device.type == "cuda":
            torch.cuda.empty_cache()
        cudnn.enabled = True
        cudnn.benchmark = True
        print(f"Device: {device}")

        os.makedirs(uploads_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        _startup_done = True


@app.after_request
def apply_security_headers(response):
    return add_security_headers(response)


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return json_response({"error": error.message}, status=error.status_code)


@app.errorhandler(413)
def handle_request_entity_too_large(_error):
    return json_response(
        {"error": f"上传内容过大，单次请求不能超过 {DEFAULT_MAX_UPLOAD_MB} MB"},
        status=413,
    )


@app.errorhandler(500)
def handle_internal_error(error):
    print(f"内部错误: {error}")
    return json_response({"error": "服务器处理失败，请稍后重试"}, status=500)


def calculate(input_point, timestamp, fx, fy, cx, cy):
    blend_image_path = ""
    cloud_path_local = ""

    image, old_img, orig_h, orig_w = calc.load_image(input_path, timestamp)
    if image is None:
        return {"error": "读取图像失败"}, blend_image_path, cloud_path_local

    seg_img, _ = calc.generate_original_mask(
        image_path=os.path.join(input_path, f"{timestamp}.jpg"),
        input_point=input_point,
        orig_w=orig_w,
        orig_h=orig_h,
        output_path=output_path,
        timestamp=timestamp,
    )

    threshold_mask, _ = calc.process_threshold_mask(seg_img, output_path, timestamp)

    depth_data = calc.load_depth_data(input_path, timestamp, H, W)
    if depth_data is None:
        return {"error": "读取深度数据失败"}, blend_image_path, cloud_path_local

    if depth_data.max() == 0 or not np.isfinite(depth_data).all():
        print("错误: 深度数据无效，无法生成点云")
        return {"error": "深度数据无效，请确认输入数据完整"}, blend_image_path, cloud_path_local

    camera_img = calc.load_camera_image(input_path, timestamp, depth_data.shape)
    if camera_img is None:
        return {"error": "读取相机图像失败"}, blend_image_path, cloud_path_local

    initial_cloud = calc.create_point_cloud(
        camera_img=camera_img,
        depth_data=depth_data,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scale=scale,
        output_path=output_path,
        model_type=model_type,
        timestamp=timestamp,
    )
    if initial_cloud is None:
        return {"error": "生成点云失败"}, blend_image_path, cloud_path_local

    valid_contours, all_contours = calc.extract_contours(threshold_mask)

    height_map, valid_indices = calc.calculate_tree_heights(
        valid_contours=valid_contours,
        all_contours=all_contours,
        threshold_mask=threshold_mask,
        depth_shape=depth_data.shape,
        initial_cloud=initial_cloud,
    )

    updated_height_map, blend_image_path = calc.generate_blend_annotated_image(
        old_img=old_img,
        threshold_mask=threshold_mask,
        all_contours=all_contours,
        height_map=height_map,
        output_path=output_path,
        timestamp=timestamp,
    )

    cloud_path_local = calc.save_point_cloud(
        initial_cloud=initial_cloud,
        valid_indices=valid_indices,
        output_path=output_path,
        model_type=model_type,
        timestamp=timestamp,
    )

    return updated_height_map, blend_image_path, cloud_path_local


def _parse_legacy_request():
    ensure_post_request(flask.request)

    uploads = ordered_uploaded_files(flask.request)
    if len(uploads) < 12:
        raise ValidationError("上传文件不完整，至少需要 12 个表单文件")

    written_files = []
    try:
        timestamp = image_timestamp_from_upload(uploads[0])

        image_path = os.path.join(input_path, f"{timestamp}.jpg")
        save_verified_image(uploads[0], image_path)
        written_files.append(image_path)

        for index, suffix in LEGACY_DATA_SUFFIXES.items():
            target_path = os.path.join(input_path, f"{timestamp}{suffix}")
            save_binary_upload(uploads[index], target_path, allowed_extensions=TEXT_EXTENSIONS)
            written_files.append(target_path)

        fx = parse_finite_float(uploads[4].filename, "fx", min_value=1e-6)
        fy = parse_finite_float(uploads[5].filename, "fy", min_value=1e-6)
        cx = parse_finite_float(uploads[6].filename, "cx", min_value=0.0)
        cy = parse_finite_float(uploads[7].filename, "cy", min_value=0.0)
        touch_x1 = parse_finite_float(uploads[8].filename, "touchX1", min_value=0.0)
        touch_y1 = parse_finite_float(uploads[9].filename, "touchY1", min_value=0.0)
        touch_x2 = parse_finite_float(uploads[10].filename, "touchX2", min_value=0.0)
        touch_y2 = parse_finite_float(uploads[11].filename, "touchY2", min_value=0.0)
    except Exception:
        cleanup_files(written_files)
        raise

    return {
        "timestamp": timestamp,
        "fx": fx * scale,
        "fy": fy * scale,
        "cx": cx * scale,
        "cy": cy * scale,
        "input_point": [[int(touch_x1), int(touch_y1)], [int(touch_x2), int(touch_y2)]],
    }


def _store_point_cloud_context(timestamp, cloud_path_local):
    if cloud_path_local and os.path.exists(cloud_path_local):
        flask.session["last_point_cloud_timestamp"] = timestamp
        flask.session["last_point_cloud_model"] = model_type
        return flask.url_for("get_point_cloud", timestamp=timestamp)

    flask.session.pop("last_point_cloud_timestamp", None)
    flask.session.pop("last_point_cloud_model", None)
    return ""


def _write_response_snapshot(response_data):
    atomic_write_text("response_data.json", json.dumps(response_data, ensure_ascii=False))


@app.route("/getData", methods=["GET", "POST"])
def handle_request():
    request_data = _parse_legacy_request()
    timestamp = request_data["timestamp"]
    print(f"\n收到请求: request_id={timestamp}")
    print(f"接收坐标: {request_data['input_point']}")

    height, blend_image_path, cloud_path_local = calculate(
        request_data["input_point"],
        timestamp,
        request_data["fx"],
        request_data["fy"],
        request_data["cx"],
        request_data["cy"],
    )

    mask = ""
    if blend_image_path and os.path.exists(blend_image_path):
        mask = image_util.image_to_base64(blend_image_path)

    point_cloud_url = _store_point_cloud_context(timestamp, cloud_path_local)

    response_item = {
        "heights": height,
        "mask": mask,
        "request_id": timestamp,
    }
    if point_cloud_url:
        response_item["point_cloud_url"] = point_cloud_url

    response_data = [response_item]
    _write_response_snapshot(response_data)
    return json_response(response_data)


def _resolve_point_cloud_from_request():
    timestamp = flask.request.args.get("timestamp", "").strip()
    if timestamp:
        candidate = build_point_cloud_path(output_path, model_type, timestamp)
        if os.path.exists(candidate):
            return candidate
        raise ValidationError("请求对应的点云文件不存在", status_code=404)

    session_timestamp = flask.session.get("last_point_cloud_timestamp")
    session_model = flask.session.get("last_point_cloud_model")
    if session_timestamp and session_model == model_type:
        candidate = build_point_cloud_path(output_path, model_type, session_timestamp)
        if os.path.exists(candidate):
            return candidate

    raise ValidationError("缺少 request_id，请先调用 /getData 获取 point_cloud_url", status_code=400)


@app.route("/get_point_cloud")
def get_point_cloud():
    point_cloud_path = _resolve_point_cloud_from_request()
    return flask.send_file(
        point_cloud_path,
        as_attachment=True,
        download_name=os.path.basename(point_cloud_path),
        conditional=False,
        etag=False,
        max_age=0,
    )


def remove_files(original_mask_path, threshold_mask_path, blend_image_path):
    cleanup_files([original_mask_path, threshold_mask_path, blend_image_path])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81, debug=False, threaded=True)
