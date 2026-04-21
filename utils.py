"""
包含了一系列计算机视觉与深度学习任务中常用的工具函数和类，主要用于数据处理、模型参数统计、指标计算等辅助功能
"""

import argparse
import numpy as np
import cv2

def str2bool(v):
    """
    功能：将字符串参数转为布尔值
    用途：解决 argparse 无法直接解析布尔类型参数的问题
    参数：v (str/int): 输入值，可以是字符串（如 "true", "false"）或数字（1, 0）
    """
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_params(model):
    """
    功能：计算模型中可训练参数的总数
    用途：快速查看模型复杂度，判断是否适合当前硬件
    参数：model: PyTorch 模型实例
    返回：可训练参数数量（int）
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_image(path):
    """
    功能：读取图像并转换为 RGB 格式
    用途：统一图像读取流程，为后续处理做准备
    参数：path (str): 图像文件路径
    返回：归一化后的 RGB 图像数组（0-1 范围）
    处理流程：
        1.使用 OpenCV 读取图像
        2.灰度图转 BGR
        3.BGR 转 RGB
        4.归一化到 [0, 1]
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

def write_depth(path, depth, bits=1):
    """
    功能：将深度图保存为 PNG 文件
    用途：可视化深度预测结果
    参数：path (str): 输出文件路径（不含扩展名）
        depth: 深度图数据
        bits (int): 位深度（1 对应 8 位，2 对应 16 位）
    处理流程：
        1.计算深度图的最小值和最大值
        2.归一化到 [0, 255] 或 [0, 65535]
        3.保存为 PNG 文件
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"))

    return

def align_depth(original,predicted):
    """
    功能：对齐原始深度图和预测深度图
    用途：消除预测深度图的尺度和偏移差异
    参数：original: 原始深度图
        predicted: 预测深度图
    返回：对齐后的深度图
    处理方法：
        1.使用最小二乘法求解线性变换 y = s*x + t
        2.其中 x 是原始深度，y 是预测深度
        3.对齐公式：aligned = (predicted - t) / s
    """
    if original.shape !=predicted.shape:
        raise ValueError(f"Shape of Original Image = {original.shape} does not align with shape of predicted image: {predicted.shape}")
    x = original.copy().flatten()
    y = predicted.copy().flatten()
    A = np.vstack([x, np.ones(len(x))]).T

    s,t = np.linalg.lstsq(A,y, rcond=None)[0]

    aligned_image = (predicted -t)/s

    return aligned_image

def img_to_patch(x, patch_size, flatten_channel=True):
    """
    功能：将图像分割为 patches
    用途：为 Vision Transformer 等模型准备输入
    参数：
    x: 输入图像张量 (B, C, H, W)
        patch_size (int): 每个 patch 的大小
        flatten_channel (bool): 是否将通道维度展平
    返回：分割后的 patches 张量
    处理流程：
        1.重塑张量形状
        2.调整维度顺序
        3.展平 patches
    """
    B,C,H,W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0,2,4,1,3,5)
    x = x.flatten(1,2)
    if flatten_channel:
        x = x.flatten(2,4)
    return x

class AverageMeter(object):
    """
    功能：计算并存储平均值和当前值
    用途：训练过程中跟踪损失、准确率等指标
    方法：
        __init__: 初始化
        reset: 重置所有统计值
        update: 更新统计值
    属性：
        val: 当前值
        avg: 平均值
        sum: 总和
        count: 计数
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
