# -*- coding:UTF-8 -*-
"""
使用 SLAM3R 进行三维重建的计算模块
替代原有的基于手机深度传感器的方案
"""

import os
import copy
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import torch

import image_util
from seg import generate_mask

# SLAM3R 集成标志
SLAM3R_AVAILABLE = False
try:
    import slam3r_integration
    SLAM3R_AVAILABLE = slam3r_integration.check_slam3r_available()
except ImportError:
    print("警告: slam3r_integration 模块不可用")


def init_slam3r(device='cuda'):
    """
    初始化 SLAM3R 模型
    """
    global SLAM3R_AVAILABLE
    if SLAM3R_AVAILABLE:
        return slam3r_integration.init_slam3r_models(device)
    return False


def load_image(input_path, timestamp):
    """
    读取并验证输入图像的有效性
    """
    image_path = os.path.join(input_path, f"{timestamp}.jpg")
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return None, None, None, None

    image = Image.open(image_path)
    old_img = copy.deepcopy(image)
    orig_h, orig_w = np.array(image).shape[:2]
    return image, old_img, orig_h, orig_w


def generate_original_mask(image_path, input_point, orig_w, orig_h, output_path, timestamp):
    """
    生成原始分割掩码并保存
    """
    original_mask_path = os.path.join(output_path, f"{timestamp}_original_mask.png")
    try:
        mask = generate_mask(image_path, input_point)
        seg_img = Image.fromarray(mask).resize((orig_w, orig_h))
        seg_img.save(original_mask_path)
    except Exception as e:
        print(f"分割错误: {e}")
        seg_img = Image.new('L', (orig_w, orig_h), 0)
        seg_img.save(original_mask_path)
    return seg_img, original_mask_path


def process_threshold_mask(seg_img, output_path, timestamp):
    """
    对原始掩码进行阈值二值化和形态学优化
    """
    seg_np = np.asarray(seg_img)
    ret, threshold_mask = cv2.threshold(
        src=seg_np,
        thresh=125,
        maxval=255,
        type=cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=4)

    threshold_mask_path = os.path.join(output_path, f"{timestamp}_threshold_mask.png")
    cv2.imwrite(threshold_mask_path, threshold_mask)
    return threshold_mask, threshold_mask_path


def create_point_cloud_slam3r(image_path, output_path, timestamp):
    """
    使用 SLAM3R 从单张图像创建点云
    """
    global SLAM3R_AVAILABLE
    
    if not SLAM3R_AVAILABLE:
        print("错误：SLAM3R 不可用")
        return None, None
    
    try:
        # 使用 SLAM3R 进行三维重建
        pcd = slam3r_integration.reconstruct_single_image_with_context(
            main_image_path=image_path,
            context_images=None,
            output_path=output_path,
            conf_threshold=1.0
        )
        
        if pcd is None or len(pcd.points) == 0:
            print("SLAM3R 生成点云失败或点云为空")
            return None, None
        
        # 保存点云
        ply_path = os.path.join(output_path, f"SLAM3R_{timestamp}_Depth.ply")
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
        print(f"SLAM3R 点云已保存: {ply_path}, 点数: {len(pcd.points)}")
        
        return pcd, pcd.points.shape if hasattr(pcd.points, 'shape') else (len(pcd.points), 3)
        
    except Exception as e:
        print(f"SLAM3R 点云创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_point_cloud_from_multiple_images(image_paths, output_path, timestamp):
    """
    使用 SLAM3R 从多张图像创建点云（效果更好）
    """
    global SLAM3R_AVAILABLE
    
    if not SLAM3R_AVAILABLE:
        print("错误：SLAM3R 不可用")
        return None
    
    try:
        pcd, ply_path = slam3r_integration.reconstruct_from_images(
            image_paths=image_paths,
            output_path=output_path,
            conf_threshold=1.5,
            num_points_save=500000
        )
        return pcd
        
    except Exception as e:
        print(f"SLAM3R 多图像点云创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_contours(threshold_mask):
    """
    从阈值掩码中提取并筛选有效轮廓
    """
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_contours = []

    mask_h = threshold_mask.shape[0]
    for orig_idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if y > mask_h * 4 / 5 or y + h < mask_h * 1 / 5:
            continue
        if cv2.contourArea(contour) > 0:
            valid_contours.append((orig_idx, contour))

    return valid_contours, contours


def calculate_tree_heights_slam3r(valid_contours, all_contours, threshold_mask, 
                                   initial_cloud, image_shape):
    """
    使用 SLAM3R 点云计算树木高度
    由于 SLAM3R 生成的点云是 224x224 尺寸，需要做尺寸映射
    """
    height_map = {}
    total_valid_indices = []
    
    total_points = len(initial_cloud.points)
    if total_points == 0:
        print("错误：点云为空，无法计算高度")
        return height_map, total_valid_indices
    
    print(f"点云总点数: {total_points}")
    
    # SLAM3R 输出的点云对应 224x224 的图像
    slam3r_size = 224
    mask_h, mask_w = threshold_mask.shape[:2]
    
    for orig_idx, contour in valid_contours:
        print(f"-----------计算第 {orig_idx} 个轮廓的高度-----------------")
        
        # 生成当前轮廓的局部掩码
        local_mask = threshold_mask.copy()
        for j in range(len(all_contours)):
            if j != orig_idx:
                cv2.fillPoly(local_mask, [all_contours[j]], 0)

        if local_mask.max() == 0:
            continue
        
        # 将掩码调整到 SLAM3R 点云尺寸 (224x224)
        # 需要先做中心裁剪，然后 resize
        min_dim = min(mask_h, mask_w)
        left = (mask_w - min_dim) // 2
        top = (mask_h - min_dim) // 2
        
        # 裁剪掩码
        cropped_mask = local_mask[top:top+min_dim, left:left+min_dim]
        
        # 调整到 224x224
        resized_mask = cv2.resize(cropped_mask, (slam3r_size, slam3r_size))
        
        # 提取有效点索引
        mask_flat = resized_mask.flatten()
        valid_indices = [k for k, val in enumerate(mask_flat) if val == 255]

        if not valid_indices:
            print(f"轮廓 {orig_idx}: 没有有效像素点")
            continue
        
        # 过滤超出点云范围的索引
        valid_indices = [idx for idx in valid_indices if idx < total_points]
        if not valid_indices:
            print(f"轮廓 {orig_idx}: 过滤后没有有效索引")
            continue
        
        print(f"轮廓 {orig_idx}: 有效索引数量 = {len(valid_indices)}")

        try:
            # 筛选当前树木的点云
            tree_cloud = image_util.display_inlier_outlier(initial_cloud, valid_indices)
            
            if len(tree_cloud.points) == 0:
                print(f"轮廓 {orig_idx}: 筛选后点云为空")
                continue
            
            # 去噪
            if len(tree_cloud.points) > 10:
                cl, denoise_ind = tree_cloud.remove_radius_outlier(nb_points=5, radius=0.5)
                if len(denoise_ind) > 0:
                    denoised_cloud = tree_cloud.select_by_index(denoise_ind)
                else:
                    denoised_cloud = tree_cloud
            else:
                denoised_cloud = tree_cloud
            
            if len(denoised_cloud.points) == 0:
                print(f"轮廓 {orig_idx}: 去噪后点云为空")
                continue

            # 计算 AABB 包围盒高度
            aabb_length = image_util.aabb(denoised_cloud)
            # SLAM3R 的尺度可能需要校准，这里假设 y 轴是高度
            tree_height = round(aabb_length[1], 2)
            print(f"轮廓 {orig_idx} 高度: {tree_height}m")

            height_map[orig_idx] = tree_height
            total_valid_indices.extend(valid_indices)
            
        except Exception as e:
            print(f"计算轮廓 {orig_idx} 高度失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    return height_map, total_valid_indices


def generate_blend_annotated_image(old_img, threshold_mask, all_contours, height_map, output_path, timestamp):
    """
    生成带高度标注的混合图像
    """
    colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128)]

    mask_img = Image.fromarray(threshold_mask).convert('RGB')
    annotate_mask = np.array(mask_img, dtype=np.uint8, copy=True)
    annotate_mask.setflags(write=True)

    color_idx = 0
    for contour_idx in range(len(all_contours)):
        if contour_idx in height_map:
            cv2.fillPoly(annotate_mask, [all_contours[contour_idx]], colors[color_idx % len(colors)])
            color_idx += 1
        else:
            cv2.fillPoly(annotate_mask, [all_contours[contour_idx]], (0, 0, 0))

    updated_height_map = {}
    color_idx = 0
    for contour_idx in range(len(all_contours)):
        if contour_idx not in height_map:
            continue

        x, y, w, h = cv2.boundingRect(all_contours[contour_idx])
        if not (x == 0 and y == 0 and w == annotate_mask.shape[1] and h == annotate_mask.shape[0]):
            cv2.rectangle(annotate_mask, (x, y), (x + w, y + h), 
                         colors[color_idx % len(colors)], thickness=3)

            tree_height = height_map[contour_idx]
            updated_height_map[color_idx] = tree_height
            text_pos = (x, y - 20) if y > 50 else (x, y + h + 42)
            cv2.putText(annotate_mask, f"height{color_idx} {tree_height}m", text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
            color_idx += 1

    blend_img = Image.blend(old_img, Image.fromarray(annotate_mask), alpha=0.7)
    blend_path = os.path.join(output_path, f"{timestamp}_blend_mask.png")
    blend_img.save(blend_path)

    return updated_height_map, blend_path


def save_point_cloud(initial_cloud, valid_indices, output_path, model_type, timestamp):
    """
    保存筛选后的树木点云
    """
    try:
        if len(initial_cloud.points) == 0:
            print("错误：点云为空，无法保存。")
            return None

        if valid_indices:
            tree_cloud = image_util.display_inlier_outlier(initial_cloud, valid_indices)
        else:
            tree_cloud = initial_cloud

        if len(tree_cloud.points) > 10:
            cl, denoise_ind = tree_cloud.remove_radius_outlier(nb_points=5, radius=0.5)
            if len(denoise_ind) > 0:
                final_cloud = tree_cloud.select_by_index(denoise_ind)
            else:
                final_cloud = tree_cloud
        else:
            final_cloud = tree_cloud

        cloud_path = os.path.join(output_path, f"{model_type}_{timestamp}_Depth_tree.ply")
        o3d.io.write_point_cloud(Path(cloud_path), final_cloud, write_ascii=True)
        return cloud_path
        
    except Exception as e:
        print(f"保存最终点云失败: {e}")
        return None
