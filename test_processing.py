# 测试后端处理流程
import numpy as np
import cv2
from PIL import Image
import calculate as calc
import image_util

# 使用有效的深度数据进行测试
timestamp = "1765801316099"  # 这个有有效深度数据
input_path = "uploads/input"
output_path = "uploads/output"
H, W = 90, 160
scale = 90 / 1080
fx = fy = 1312.1901 * scale
cx = 542.00885 * scale
cy = 966.3707 * scale

print("=" * 50)
print("测试后端处理流程")
print("=" * 50)

# 1. 读取图像
print("\n1. 读取图像...")
image, old_img, orig_h, orig_w = calc.load_image(input_path, timestamp)
if image is None:
    print("错误：无法读取图像")
    exit(1)
print(f"   图像尺寸: {orig_w}x{orig_h}")

# 2. 读取深度数据
print("\n2. 读取深度数据...")
depth_data = calc.load_depth_data(input_path, timestamp, H, W)
if depth_data is None:
    print("错误：无法读取深度数据")
    exit(1)
print(f"   深度数据尺寸: {depth_data.shape}")
print(f"   深度数据范围: min={depth_data.min()}, max={depth_data.max()}")

if depth_data.max() == 0:
    print("错误：深度数据全为0")
    exit(1)

# 3. 读取相机图像
print("\n3. 读取相机图像...")
camera_img = calc.load_camera_image(input_path, timestamp, depth_data.shape)
if camera_img is None:
    print("错误：无法读取相机图像")
    exit(1)
print(f"   相机图像尺寸: {camera_img.shape}")

# 4. 创建点云
print("\n4. 创建点云...")
initial_cloud = calc.create_point_cloud(
    camera_img=camera_img,
    depth_data=depth_data,
    fx=fx, fy=fy, cx=cx, cy=cy,
    scale=scale,
    output_path=output_path,
    model_type="Test",
    timestamp=timestamp
)
if initial_cloud is None:
    print("错误：无法创建点云")
    exit(1)
print(f"   点云点数: {len(initial_cloud.points)}")

print("\n" + "=" * 50)
print("测试完成！后端处理流程正常工作。")
print("=" * 50)
