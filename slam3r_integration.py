# -*- coding:UTF-8 -*-
"""
SLAM3R 集成模块
用于将 SLAM3R 模型应用到树木高度测量系统中
SLAM3R 可以从单目 RGB 图像/视频进行三维重建，无需深度传感器
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d

# 添加 SLAM3R 路径
SLAM3R_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SLAM3R-main')
sys.path.insert(0, SLAM3R_PATH)

# 全局模型变量
_i2p_model = None
_l2w_model = None
_device = None


def init_slam3r_models(device=None):
    """
    初始化 SLAM3R 模型
    :param device: 设备 ('cuda' 或 'cpu')
    :return: 是否初始化成功
    """
    global _i2p_model, _l2w_model, _device
    
    if _i2p_model is not None and _l2w_model is not None:
        print("SLAM3R 模型已初始化")
        return True
    
    try:
        from slam3r.models import Image2PointsModel, Local2WorldModel
        import torch
        
        # 强制使用 CUDA
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"SLAM3R 使用 CUDA: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("警告: SLAM3R - CUDA 不可用，将使用 CPU")
        
        _device = device
        print(f"正在加载 SLAM3R 模型到 {device}...")
        
        # 从 HuggingFace 加载预训练模型
        print("加载 Image2Points 模型...")
        _i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        _i2p_model.to(device)
        _i2p_model.eval()
        
        print("加载 Local2World 模型...")
        _l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
        _l2w_model.to(device)
        _l2w_model.eval()
        
        print("SLAM3R 模型加载完成")
        return True
        
    except Exception as e:
        print(f"SLAM3R 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_image(image_path, size=224):
    """
    预处理图像为 SLAM3R 输入格式
    :param image_path: 图像路径
    :param size: 目标尺寸 (SLAM3R 要求 224x224)
    :return: 预处理后的图像字典
    """
    from slam3r.utils.image import ImgNorm
    
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    
    # 计算裁剪区域（中心裁剪为正方形）
    min_dim = min(W, H)
    left = (W - min_dim) // 2
    top = (H - min_dim) // 2
    img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
    
    # 调整到目标尺寸
    img_resized = img_cropped.resize((size, size), Image.LANCZOS)
    img_np = np.array(img_resized)
    
    # 应用归一化
    img_tensor = ImgNorm(img_np).unsqueeze(0)  # (1, 3, H, W)
    
    return {
        'img': img_tensor,
        'true_shape': torch.tensor([[size, size]]),
        'idx': 0,
        'instance': image_path,
        'original_size': (W, H),
        'crop_params': (left, top, min_dim)
    }


def preprocess_images_batch(image_paths, size=224):
    """
    批量预处理多张图像
    :param image_paths: 图像路径列表
    :param size: 目标尺寸
    :return: 预处理后的图像字典列表
    """
    views = []
    for idx, path in enumerate(image_paths):
        view = preprocess_image(path, size)
        view['idx'] = idx
        views.append(view)
    return views


@torch.no_grad()
def reconstruct_from_images(image_paths, output_path, 
                            conf_threshold=1.5, 
                            num_points_save=500000):
    """
    从多张图像进行三维重建
    :param image_paths: 图像路径列表（至少3张，建议5-10张）
    :param output_path: 输出路径
    :param conf_threshold: 置信度阈值
    :param num_points_save: 保存的点数
    :return: 点云对象和保存路径
    """
    global _i2p_model, _l2w_model, _device
    
    if _i2p_model is None or _l2w_model is None:
        if not init_slam3r_models(_device or 'cuda'):
            return None, None
    
    from slam3r.utils.device import to_numpy, collate_with_cat
    from slam3r.utils.image import rgb
    
    print(f"开始 SLAM3R 三维重建，共 {len(image_paths)} 张图像...")
    
    # 预处理图像
    views = preprocess_images_batch(image_paths)
    
    # 将图像移动到设备
    for view in views:
        view['img'] = view['img'].to(_device)
        view['true_shape'] = view['true_shape'].to(_device)
    
    # 使用 I2P 模型生成点云
    print("生成初始点云...")
    ref_id = len(views) // 2  # 使用中间帧作为参考帧
    preds = _i2p_model(views, ref_id=ref_id)
    
    # 收集点云数据
    all_pts = []
    all_colors = []
    all_confs = []
    
    for i, (view, pred) in enumerate(zip(views, preds)):
        # 获取点云坐标
        if 'pts3d' in pred:
            pts3d = to_numpy(pred['pts3d'][0])  # (H, W, 3)
        elif 'pts3d_in_other_view' in pred:
            pts3d = to_numpy(pred['pts3d_in_other_view'][0])
        else:
            continue
        
        # 获取置信度
        conf = to_numpy(pred['conf'][0])  # (H, W)
        
        # 获取颜色
        img_np = rgb(view['img'][0])  # (H, W, 3)
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # 展平
        pts_flat = pts3d.reshape(-1, 3)
        colors_flat = img_np.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        
        # 过滤低置信度点
        valid_mask = conf_flat > conf_threshold
        pts_valid = pts_flat[valid_mask]
        colors_valid = colors_flat[valid_mask]
        conf_valid = conf_flat[valid_mask]
        
        all_pts.append(pts_valid)
        all_colors.append(colors_valid)
        all_confs.append(conf_valid)
        
        print(f"  帧 {i}: 有效点数 = {len(pts_valid)}")
    
    # 合并所有点云
    if not all_pts:
        print("错误：没有生成有效的点云")
        return None, None
    
    merged_pts = np.concatenate(all_pts, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    
    print(f"合并后点云总点数: {len(merged_pts)}")
    
    # 采样点云（如果点数太多）
    if len(merged_pts) > num_points_save:
        indices = np.random.choice(len(merged_pts), num_points_save, replace=False)
        merged_pts = merged_pts[indices]
        merged_colors = merged_colors[indices]
        print(f"采样后点云点数: {len(merged_pts)}")
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors / 255.0)
    
    # 保存点云
    os.makedirs(output_path, exist_ok=True)
    ply_path = os.path.join(output_path, 'slam3r_reconstruction.ply')
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
    print(f"点云已保存到: {ply_path}")
    
    return pcd, ply_path


@torch.no_grad()
def reconstruct_single_image_with_context(main_image_path, context_images=None, 
                                          output_path=None, conf_threshold=1.0):
    """
    使用单张主图像进行重建（可选上下文图像辅助）
    这是针对当前系统的适配：用户上传单张图像，系统使用 SLAM3R 生成点云
    
    :param main_image_path: 主图像路径
    :param context_images: 可选的上下文图像路径列表
    :param output_path: 输出路径
    :param conf_threshold: 置信度阈值
    :return: 点云对象
    """
    global _i2p_model, _device
    
    if _i2p_model is None:
        if not init_slam3r_models(_device or 'cuda'):
            return None
    
    from slam3r.utils.device import to_numpy
    from slam3r.utils.image import rgb
    
    print(f"SLAM3R 单图像重建: {main_image_path}")
    
    # 准备输入视图
    if context_images and len(context_images) >= 2:
        # 使用上下文图像
        all_images = context_images[:2] + [main_image_path] + context_images[2:4]
        views = preprocess_images_batch(all_images[:5])  # 最多5帧
        ref_id = len(views) // 2
    else:
        # 仅使用单张图像 - 通过复制创建伪多视图
        # 注意：这种方式效果有限，但可以生成基本的深度估计
        view = preprocess_image(main_image_path)
        views = [view]
        ref_id = 0
    
    # 移动到设备
    for view in views:
        view['img'] = view['img'].to(_device)
        view['true_shape'] = view['true_shape'].to(_device)
    
    # 生成点云
    preds = _i2p_model(views, ref_id=ref_id)
    
    # 处理参考帧的点云
    pred = preds[ref_id]
    if 'pts3d' in pred:
        pts3d = to_numpy(pred['pts3d'][0])  # (H, W, 3)
    else:
        print("错误：无法获取点云数据")
        return None
    
    conf = to_numpy(pred['conf'][0])
    img_np = rgb(views[ref_id]['img'][0])
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # 展平和过滤
    pts_flat = pts3d.reshape(-1, 3)
    colors_flat = img_np.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    
    valid_mask = conf_flat > conf_threshold
    pts_valid = pts_flat[valid_mask]
    colors_valid = colors_flat[valid_mask]
    
    print(f"生成点云点数: {len(pts_valid)}")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid / 255.0)
    
    # 保存点云
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        ply_path = os.path.join(output_path, 'slam3r_single.ply')
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
        print(f"点云已保存到: {ply_path}")
    
    return pcd


def get_depth_from_pointcloud(pcd, image_shape, intrinsics=None):
    """
    从点云生成深度图（用于兼容现有系统）
    :param pcd: Open3D 点云对象
    :param image_shape: 目标深度图尺寸 (H, W)
    :param intrinsics: 相机内参 (fx, fy, cx, cy)
    :return: 深度图 numpy 数组
    """
    if pcd is None or len(pcd.points) == 0:
        return np.zeros(image_shape, dtype=np.uint16)
    
    points = np.asarray(pcd.points)
    H, W = image_shape
    
    # 如果没有内参，使用默认值
    if intrinsics is None:
        fx = fy = W  # 假设焦距约等于图像宽度
        cx, cy = W / 2, H / 2
    else:
        fx, fy, cx, cy = intrinsics
    
    # 投影点云到图像平面
    depth_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.int32)
    
    for pt in points:
        x, y, z = pt
        if z <= 0:
            continue
        
        # 投影
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        
        if 0 <= u < W and 0 <= v < H:
            depth_map[v, u] += z
            count_map[v, u] += 1
    
    # 平均深度
    valid_mask = count_map > 0
    depth_map[valid_mask] /= count_map[valid_mask]
    
    # 转换为毫米单位的 uint16
    depth_mm = (depth_map * 1000).astype(np.uint16)
    
    return depth_mm


def check_slam3r_available():
    """
    检查 SLAM3R 是否可用
    """
    try:
        from slam3r.models import Image2PointsModel, Local2WorldModel
        return True
    except ImportError as e:
        print(f"SLAM3R 不可用: {e}")
        return False


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("SLAM3R 集成模块测试")
    print("=" * 50)
    
    # 检查可用性
    if not check_slam3r_available():
        print("请先安装 SLAM3R 依赖")
        exit(1)
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    if init_slam3r_models(device):
        print("模型初始化成功")
    else:
        print("模型初始化失败")
        exit(1)
    
    # 测试单图像重建
    test_image = "uploads/input/1765801316099.jpg"
    if os.path.exists(test_image):
        pcd = reconstruct_single_image_with_context(
            test_image, 
            output_path="uploads/output"
        )
        if pcd:
            print(f"测试成功，生成点云点数: {len(pcd.points)}")
    else:
        print(f"测试图像不存在: {test_image}")
