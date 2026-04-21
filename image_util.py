"""
是一个图像处理与点云处理的工具类代码，包含了一系列用于图像预处理、深度图处理、
点云生成与滤波、坐标转换等功能的函数，依赖PIL、OpenCV、Open3D、numpy等库，
适用于计算机视觉中涉及深度估计、点云处理的场景。
"""
import base64

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


# 相机内参
# cam = joblib.load('mtx.pkl')

def Rel2Abs(x_rel, y_abs, n):
    '''
    将相对深度图调整到绝对深度图的值域
    x_rel:相对深度
    y_abs:绝对深度
    n：最高次数，阶数
    '''
    # collapsed into one dimension
    y = y_abs.copy().flatten()  # Absolute Depth
    x = x_rel.copy().flatten()  # Relative Depth
    #     A = np.vstack([x, np.ones(len(x))]).T
    #     s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    #     return x_rel*s+t
    p = np.poly1d(np.polyfit(x, y, n))  # 拟合并构造出一个n阶多项式
    depth_aligned = 0.0
    for c in p.coeffs:
        depth_aligned *= x_rel
        depth_aligned += c
    return depth_aligned


def letterbox_image(image, size):
    '''
    为 RGB 图像添加灰条以调整至目标尺寸（size）
    添加灰条，修改图片尺寸
    '''
    image = image.convert("RGB")
    iw, ih = image.size  # 图片原尺寸
    w, h = size  # 需要的尺寸
    scale = min(w / iw, h / ih)  # 找到较小的倍数
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))  # 灰色底部
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh


def letterbox_depth(depth, size):
    '''
    处理单通道的灰度图，同样修改图片尺寸
    '''
    iw, ih = depth.size  # 图片原尺寸
    w, h = size  # 需要的尺寸
    scale = min(w / iw, h / ih)  # 找到较小的倍数
    nw = int(iw * scale)
    nh = int(ih * scale)

    depth = depth.resize((nw, nh), Image.BICUBIC)
    new_depth = Image.new('L', size, 0)
    new_depth.paste(depth, ((w - nw) // 2, (h - nh) // 2))
    return new_depth, nw, nh


def CreatePointCloud(img, depth, fx, fy, cx, cy, scale, depthScale, fileName):
    '''
    根据 RGB 图（img）和深度图（depth）生成点云。
    需输入相机内参（fx, fy, cx, cy，焦距和光学中心）、
    缩放比例（scale）、深度尺度（depthScale，深度值与米的比值），
    生成点云后进行翻转校正并保存为指定文件（fileName），返回点云对象
    img:RGB图
    depth：深度图 类型只能是uint8 uint16 float 若img和depth大小不一样会报错
    scale:img相对于相机获取的原始图像，被放缩的比例
    depthScale:深度图的数值和米的比值
    depthTrunc:大于这个值的深度看做0
    fileName:生成的点云的文件名
    '''
    height, width, _ = img.shape
    depth_h, depth_w = depth.shape[:2]
    
    print(f"CreatePointCloud - 图像尺寸: {img.shape}, 深度图尺寸: {depth.shape}")
    print(f"CreatePointCloud - 深度范围: min={depth.min()}, max={depth.max()}")
    print(f"CreatePointCloud - 相机内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # 检查深度数据是否有效
    if depth.max() == 0:
        print("警告：深度数据全为0，无法生成有效点云")
        # 返回空点云而不是崩溃
        pcd = o3d.geometry.PointCloud()
        return pcd
    
    # 检查尺寸是否匹配
    if height != depth_h or width != depth_w:
        print(f"警告：图像和深度图尺寸不匹配，调整深度图尺寸")
        depth = cv2.resize(depth, (width, height))
    
    # 确保深度图类型正确
    if depth.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        depth = depth.astype(np.uint16)

    # 焦距（fx，fy），光学中心（cx，cy）
    # 存储相机内参和图像高和宽
    cam_o3 = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # 计算depth_trunc，避免除以0
    depth_max = float(depth.max())
    if scale > 0:
        depth_trunc = depth_max / scale
    else:
        depth_trunc = depth_max
    
    # 确保depth_trunc是合理的值
    depth_trunc = max(depth_trunc, 1.0)
    print(f"CreatePointCloud - depth_trunc: {depth_trunc}")
    
    # 生成rgbd图
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(img)), 
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=depthScale, 
        depth_trunc=depth_trunc, 
        convert_rgb_to_intensity=False)

    # 生成三维点云需要rgbd图和相机内参
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_o3)
    print(f"CreatePointCloud - 生成点云点数: {len(pcd.points)}")
    
    # 翻转点云
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # 保存点云（即使为空也保存，避免后续文件不存在的问题）
    if len(pcd.points) > 0:
        o3d.io.write_point_cloud(fileName, pcd, write_ascii=True)
    else:
        print("警告：生成的点云为空，跳过保存")
    
    return pcd


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    # max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)  # 设置为True表示保存ind之外的点
    #     outlier_cloud.paint_uniform_color([0, 1, 0])
    #     inlier_cloud.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([inlier_cloud],width=600,height=600)
    return inlier_cloud


"""
半径滤波去除点云噪声。
通过判断每个点的指定半径（ball_radius）内是否包含至少minpoints个点，保留符合条件的点，返回滤波后的点云。
"""


def radius_outlier_removal(tree_cloud, minpoints, ball_radius):
    print("Radius oulier removal")
    cl, ind = tree_cloud.remove_radius_outlier(nb_points=minpoints,
                                               radius=ball_radius)  # nb_points：球体中最少点的数量 radius球的半径
    radius_cloud = tree_cloud.select_by_index(ind)
    # o3d.visualization.draw_geometries([radius_cloud], window_name="半径滤波",
    #                                   width=700, height=700,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)
    return radius_cloud


"""
计算点云中每个点到其k个最近邻的平均距离。
通过 KD 树实现近邻搜索，返回每个点的平均距离数组。
"""


def get_avg_distance(cloud, k):
    '''
    cloud：点云
    k：用来计算平均距离的k个最近邻居
    '''
    point = np.asarray(cloud.points)  # 获取点坐标
    kdtree = o3d.geometry.KDTreeFlann(cloud)  # 建立KD树索引
    point_size = point.shape[0]  # 获取点的个数
    dd = np.zeros(point_size)
    for i in range(point_size):
        [_, idx, dis] = kdtree.search_knn_vector_3d(point[i], k + 1)
        dd[i] = np.mean(np.sqrt(dis[1:]))  # 获取到k个最近邻点的平均距离 第1个为到自己的距离
    return dd


"""
删除点云中指定索引（ind）的点，可视化内点（保留）和外点（删除，标为绿色），返回保留的点云。
"""


def delete_given_points(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud = cloud.select_by_index(ind)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    return inlier_cloud


"""
统计滤波去除点云噪声。
基于邻域（neighbors个点）平均距离的标准差，保留在阈值（std_threshold）范围内的点，返回滤波后的点云。
"""


def stat_outlier_removal(radius_cloud, neighbors, std_threshold):
    print("Statistical oulier removal")
    cl, ind = radius_cloud.remove_statistical_outlier(nb_neighbors=neighbors,
                                                      std_ratio=std_threshold)
    sor_cloud = radius_cloud.select_by_index(ind)
    return sor_cloud


"""
计算点云的轴对齐包围盒（AABB），返回包围盒在 x、y、z 方向的边长。
"""


def aabb(cloud):
    aabb = cloud.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)  # aabb包围盒为红色

    # [center_x, center_y, center_z] = aabb.get_center()
    # print("aabb包围盒的中心坐标为：\n", [center_x, center_y, center_z])

    # vertex_set = np.asarray(aabb.get_box_points())
    # print("obb包围盒的顶点为：\n", vertex_set)

    aabb_box_length = np.asarray(aabb.get_extent())  # x y z
    # print("aabb包围盒的边长为：\n", aabb_box_length)

    # half_extent = np.asarray(aabb.get_half_extent())
    # print("aabb包围盒边长的一半为：\n", half_extent)

    # max_bound = np.asarray(aabb.get_max_bound())
    # print("aabb包围盒边长的最大值为：\n", max_bound)

    # max_extent = np.asarray(aabb.get_max_extent())
    # print("aabb包围盒边长的最大范围，即X, Y和Z轴的最大值：\n", max_extent)

    # min_bound = np.asarray(aabb.get_min_bound())
    # print("aabb包围盒边长的最小值为：\n", min_bound)

    # o3d.visualization.draw_geometries([cloud, aabb], window_name="AABB包围盒",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)

    return aabb_box_length


"""
将图像文件（path）转换为 base64 编码字符串，用于图像的序列化传输或存储。
"""


def image_to_base64(path):
    with open(path, 'rb') as f:
        code = base64.b64encode(f.read()).decode()
        return code


"""
笛卡尔坐标与极坐标的相互转换
"""


def cart2pol(x, y):  # 笛卡尔坐标系->极坐标系
    theta = np.arctan2(y, x)  # 求夹角
    rho = np.hypot(x, y)  # 求斜边长度，即半径
    return theta, rho


def pol2cart(theta, rho):  # 极坐标系->笛卡尔坐标系
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


"""
旋转轮廓（cnt）。
以rotatepoint为旋转中心，按angle角度旋转轮廓，通过笛卡尔坐标与极坐标的转换实现，返回旋转后的轮廓。
"""


def rotate_contour(cnt, rotatepoint, angle):  # 轮廓旋转函数
    cx = rotatepoint[0]
    cy = rotatepoint[1]
    cnt_norm = cnt - [cx, cy]  # 将轮廓移动到旋转点
    coordinates = cnt_norm[:, 0, :]  # 找到x=0的点
    xs, ys = coordinates[:, 0], coordinates[:, 1]  # 起点
    thetas, rhos = cart2pol(xs, ys)  # 起点转换成极坐标形式
    thetas = np.rad2deg(thetas)  # 弧度转角度
    thetas = (thetas + angle) % 360  # 最终旋转角
    thetas = np.deg2rad(thetas)  # 转回幅度
    xs, ys = pol2cart(thetas, rhos)  # 起点转换回笛卡尔坐标系
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys
    cnt_rotated = cnt_norm + [cx, cy]  # 轮廓平移回原来的位置
    cnt_rotated = cnt_rotated.astype(np.int32)
    return cnt_rotated
