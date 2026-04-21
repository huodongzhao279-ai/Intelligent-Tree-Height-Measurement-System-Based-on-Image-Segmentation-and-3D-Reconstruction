import os
import copy
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import image_util
from seg import generate_mask

def load_image(input_path, timestamp):
    """
    读取并验证输入图像的有效性
    :param input_path: 输入文件夹路径
    :param timestamp: 时间戳（用于拼接文件名）
    :return: 原始图像、备份图像、图像高度、图像宽度（失败返回None）
    """
    image_path = os.path.join(input_path, f"{timestamp}.jpg")
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return None, None, None, None

    image = Image.open(image_path)
    old_img = copy.deepcopy(image)  # 备份用于后续混合绘图
    orig_h, orig_w = np.array(image).shape[:2]
    return image, old_img, orig_h, orig_w


def generate_original_mask(image_path, input_point, orig_w, orig_h, output_path, timestamp):
    """
    生成原始分割掩码并保存（含异常处理）
    :param image_path: 原始图像路径
    :param input_point: 输入点（用于分割）
    :param orig_w: 原始图像宽度
    :param orig_h: 原始图像高度
    :param output_path: 输出文件夹路径
    :param timestamp: 时间戳
    :return: 分割掩码图像、掩码保存路径
    """
    original_mask_path = os.path.join(output_path, f"{timestamp}_original_mask.png")
    try:
        # 调用分割函数生成掩码
        mask = generate_mask(image_path, input_point)
        seg_img = Image.fromarray(mask).resize((orig_w, orig_h))
        seg_img.save(original_mask_path)
    except Exception as e:
        print(f"分割错误: {e}")
        # 异常时生成空白掩码
        seg_img = Image.new('L', (orig_w, orig_h), 0)
        seg_img.save(original_mask_path)
    return seg_img, original_mask_path


def process_threshold_mask(seg_img, output_path, timestamp):
    """
    对原始掩码进行阈值二值化和形态学优化
    :param seg_img: 原始分割掩码
    :param output_path: 输出文件夹路径
    :param timestamp: 时间戳
    :return: 优化后的阈值掩码、掩码保存路径
    """
    # 全局阈值二值化
    seg_np = np.asarray(seg_img)
    ret, threshold_mask = cv2.threshold(
        src=seg_np, # 要二值化的图片
        thresh=125, # 全局阈值
        maxval=255, # 大于全局阈值后设定的值
        type=cv2.THRESH_BINARY  # 设定的二值化类型
    )

    # 形态学开运算（先腐蚀后膨胀，去除小噪声）
    kernel = np.ones((3, 3), np.uint8)
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, kernel, iterations=4)

    # 保存阈值掩码
    threshold_mask_path = os.path.join(output_path, f"{timestamp}_threshold_mask.png")
    cv2.imwrite(threshold_mask_path, threshold_mask)
    return threshold_mask, threshold_mask_path


def load_depth_data(input_path, timestamp, H, W):
    """
    读取手机深度相关数据（depth/rawdepth/confidence）并优化
    :param input_path: 输入文件夹路径
    :param timestamp: 时间戳
    :param H: 深度图高度（配置项）
    :param W: 深度图宽度（配置项）
    :return: 优化后的深度数据（失败返回None）
    """
    # 读取基础深度数据
    try:
        depth_path = os.path.join(input_path, f"{timestamp}_depthdata.txt")
        depth_data = np.fromfile(depth_path, dtype=np.uint16)
        print(f"深度数据原始大小: {depth_data.shape}, 期望: {H * W}")
        
        # 检查数据大小是否匹配
        if depth_data.size != H * W:
            print(f"警告：深度数据大小不匹配，尝试自动调整")
            # 尝试不同的reshape方式
            if depth_data.size == W * H:
                depth_data = depth_data.reshape(W, H).T
            else:
                print(f"错误：无法reshape深度数据")
                return None
        else:
            depth_data = depth_data.reshape(H, W)
        
        # 旋转90度后尺寸变为 (W, H)
        depth_data = cv2.rotate(depth_data, cv2.ROTATE_90_CLOCKWISE)
        print(f"深度数据旋转后尺寸: {depth_data.shape}")
        
    except Exception as e:
        print(f"读取深度数据失败: {e}")
        return None

    # 读取原始深度和置信度数据（含备用逻辑）
    try:
        raw_depth = np.fromfile(os.path.join(input_path, f"{timestamp}_rawdepthdata.txt"), dtype=np.uint16)
        raw_depth = raw_depth.reshape(H, W)
        raw_depth = cv2.rotate(raw_depth, cv2.ROTATE_90_CLOCKWISE)

        confidence = np.fromfile(os.path.join(input_path, f"{timestamp}_confidencedata.txt"), dtype=np.uint8)
        confidence = confidence.reshape(H, W)
        confidence = cv2.rotate(confidence, cv2.ROTATE_90_CLOCKWISE)
    except Exception as e:
        print(f"读取原始深度或置信度数据失败: {e}")
        raw_depth = depth_data.copy()
        confidence = np.ones_like(depth_data, dtype=np.uint8) * 255

    # 优化深度数据：用高置信度的原始深度替换（向量化操作，更高效）
    optimized_depth = depth_data.copy()
    mask = (raw_depth > 0) & (confidence > 155)
    replace_cnt = np.sum(mask)
    optimized_depth[mask] = raw_depth[mask]
    print(f"替换了 {replace_cnt} 个像素的深度数据")
    print(f"深度数据范围: min={optimized_depth.min()}, max={optimized_depth.max()}")

    # 替换 NaN/Inf 为 0
    optimized_depth = np.nan_to_num(optimized_depth, nan=0, posinf=0, neginf=0)

    return optimized_depth.astype('uint16')


def load_camera_image(input_path, timestamp, depth_shape):
    """
    读取相机图像并调整尺寸（用于点云创建）
    :param input_path: 输入文件夹路径
    :param timestamp: 时间戳
    :param depth_shape: 深度图的shape (height, width)，用于匹配尺寸
    :return: 调整后的相机图像（失败返回None）
    """
    try:
        img_path = os.path.join(input_path, f"{timestamp}.jpg")
        camera_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if camera_img is None:
            print(f"错误：无法读取图像文件: {img_path}")
            return None
        
        # 转换BGR到RGB
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸以匹配深度图 (width, height) for cv2.resize
        target_size = (depth_shape[1], depth_shape[0])
        camera_img = cv2.resize(camera_img, target_size)
        print(f"相机图像调整后尺寸: {camera_img.shape}")
        
        return camera_img
    except Exception as e:
        print(f"读取相机图像失败: {e}")
        return None


def create_point_cloud(camera_img, depth_data, fx, fy, cx, cy, scale, output_path, model_type, timestamp):
    """
    根据图像和深度数据创建初始点云
    :param camera_img: 相机图像
    :param depth_data: 优化后的深度数据
    :param fx/fy/cx/cy: 相机内参
    :param scale: 缩放因子（配置项）
    :param output_path: 输出文件夹路径
    :param model_type: 模型类型（用于文件名）
    :param timestamp: 时间戳
    :return: 初始点云（失败返回None）
    """
    ply_path = os.path.join(output_path, f"{model_type}_{timestamp}_Depth.ply")
    try:
        return image_util.CreatePointCloud(
            img=camera_img,
            depth=depth_data,
            fx=fx, fy=fy, cx=cx, cy=cy,
            scale=scale,
            depthScale=1000,
            fileName=ply_path
        )
    except Exception as e:
        print(f"创建初始点云失败: {e}")
        return None


def extract_contours(threshold_mask):
    """
    从阈值掩码中提取并筛选有效轮廓（聚焦树木主体区域）
    :param threshold_mask: 优化后的阈值掩码
    :return: 有效轮廓列表（含原始索引）、所有轮廓
    """
    # 从二值掩码中提取所有连通区域
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    valid_contours = []

    mask_h = threshold_mask.shape[0]
    # 筛选条件：覆盖图像高度 1/5~4/5 区间（扩展范围以支持非居中树木）+ 非空面积
    for orig_idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if y > mask_h * 4 / 5 or y + h < mask_h * 1 / 5:
            continue  # 排除极端上下区域的无效轮廓
        if cv2.contourArea(contour) > 0:
            valid_contours.append((orig_idx, contour))

    return valid_contours, contours


def calculate_tree_heights(valid_contours, all_contours, threshold_mask, depth_shape, initial_cloud):
    """
    计算每个有效轮廓对应的树木高度（基于点云AABB包围盒）
    :param valid_contours: 有效轮廓列表
    :param all_contours: 所有轮廓
    :param threshold_mask: 阈值掩码
    :param depth_shape: 深度图的shape (height, width)
    :param initial_cloud: 初始点云
    :return: 轮廓索引-高度映射、所有有效点云索引
    """
    height_map = {}
    total_valid_indices = []
    
    # 获取点云总点数
    total_points = len(initial_cloud.points)
    if total_points == 0:
        print("错误：点云为空，无法计算高度")
        return height_map, total_valid_indices
    
    print(f"点云总点数: {total_points}")

    for orig_idx, contour in valid_contours:
        print(f"-----------计算第 {orig_idx} 个轮廓的高度-----------------")
        # 生成当前轮廓的局部掩码（屏蔽其他轮廓）
        local_mask = threshold_mask.copy()
        for j in range(len(all_contours)):
            if j != orig_idx:
                cv2.fillPoly(local_mask, [all_contours[j]], 0)

        if local_mask.max() == 0:
            continue  # 跳过空掩码

        # 调整掩码尺寸以匹配深度图/点云尺寸
        # resize参数是 (width, height)
        resize_mask = Image.fromarray(local_mask).resize((depth_shape[1], depth_shape[0]))
        mask_flat = np.asarray(resize_mask).flatten()
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
            # 筛选当前树木的点云并去噪
            tree_cloud = image_util.display_inlier_outlier(initial_cloud, valid_indices)
            
            if len(tree_cloud.points) == 0:
                print(f"轮廓 {orig_idx}: 筛选后点云为空")
                continue
            
            # 去噪（如果点数太少则跳过去噪）
            if len(tree_cloud.points) > 10:
                cl, denoise_ind = tree_cloud.remove_radius_outlier(nb_points=5, radius=2.0)
                if len(denoise_ind) > 0:
                    denoised_cloud = tree_cloud.select_by_index(denoise_ind)
                else:
                    denoised_cloud = tree_cloud
            else:
                denoised_cloud = tree_cloud
            
            if len(denoised_cloud.points) == 0:
                print(f"轮廓 {orig_idx}: 去噪后点云为空")
                continue

            # 计算AABB包围盒高度（y轴长度，对应树木高度）
            aabb_length = image_util.aabb(denoised_cloud)
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
    生成带高度标注的混合图像（原始图像+掩码标注）
    :param old_img: 原始图像备份
    :param threshold_mask: 阈值掩码
    :param all_contours: 所有轮廓
    :param height_map: 轮廓索引-高度映射
    :param output_path: 输出文件夹路径
    :param timestamp: 时间戳
    :return: 更新后的高度映射（连续编号）、混合图像保存路径
    """
    # 标注颜色
    colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 12)]

    # 初始化RGB掩码
    mask_img = Image.fromarray(threshold_mask).convert('RGB')
    annotate_mask = np.array(mask_img, dtype=np.uint8, copy=True)
    annotate_mask.setflags(write=True)

    # 1. 填充轮廓颜色（有效轮廓彩色，无效轮廓黑色）
    color_idx = 0
    for contour_idx in range(len(all_contours)):
        if contour_idx in height_map:
            cv2.fillPoly(annotate_mask, [all_contours[contour_idx]], colors[color_idx % len(colors)])
            color_idx += 1
        else:
            cv2.fillPoly(annotate_mask, [all_contours[contour_idx]], (0, 0, 0))

    # 2. 绘制边界框和高度文字（重新编号为连续索引）
    updated_height_map = {}
    color_idx = 0
    for contour_idx in range(len(all_contours)):
        if contour_idx not in height_map:
            continue

        x, y, w, h = cv2.boundingRect(all_contours[contour_idx])
        # 排除覆盖整个图像的异常轮廓
        if not (x == 0 and y == 0 and w == annotate_mask.shape[1] and h == annotate_mask.shape[0]):
            # 绘制边界框
            cv2.rectangle(
                annotate_mask,
                (x, y), (x + w, y + h),
                colors[color_idx % len(colors)],
                thickness=3
            )

            # 绘制高度文字（自适应位置）
            tree_height = height_map[contour_idx]
            updated_height_map[color_idx] = tree_height
            text_pos = (x, y - 20) if y > 50 else (x, y + h + 42)
            cv2.putText(
                annotate_mask,
                f"height{color_idx} {tree_height}m",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2
            )
            color_idx += 1

    # 3. 混合原始图像和标注图像（透明度70%）
    blend_img = Image.blend(old_img, Image.fromarray(annotate_mask), alpha=0.7)
    blend_path = os.path.join(output_path, f"{timestamp}_blend_mask.png")
    blend_img.save(blend_path)

    return updated_height_map, blend_path


def save_point_cloud(initial_cloud, valid_indices, output_path, model_type, timestamp):
    """
    保存筛选后的树木点云（去噪后）
    :param initial_cloud: 初始点云
    :param valid_indices: 所有有效点云索引
    :param output_path: 输出文件夹路径
    :param model_type: 模型类型
    :param timestamp: 时间戳
    :return: 最终点云保存路径（失败返回None）
    """
    try:
        # 检查点云是否为空
        if len(initial_cloud.points) == 0:
            print("错误：点云为空，无法保存。")
            return None

        # 筛选有效树木点云
        if valid_indices:
            tree_cloud = image_util.display_inlier_outlier(initial_cloud, valid_indices)
        else:
            tree_cloud = initial_cloud

        # 二次去噪
        cl, denoise_ind = tree_cloud.remove_radius_outlier(nb_points=10, radius=1.5)
        final_cloud = tree_cloud.select_by_index(denoise_ind)

        # 保存点云
        cloud_path = os.path.join(output_path, f"{model_type}_{timestamp}_Depth_tree.ply")
        o3d.io.write_point_cloud(Path(cloud_path), final_cloud, write_ascii=True)
        return cloud_path
    except Exception as e:
        print(f"保存最终点云失败: {e}")
        return None


