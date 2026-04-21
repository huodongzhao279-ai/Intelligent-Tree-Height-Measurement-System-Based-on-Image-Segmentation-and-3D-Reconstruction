"""
定义了三个用于评估图像分割模型性能的关键指标函数，适用于二值分割任务（如前景与背景分割）
"""

import numpy as np
import torch
import torch.nn.functional as F

"""
交并比（Intersection over Union）
衡量预测分割结果与真实标签的重叠程度，是分割任务中最常用的指标之一。
"""
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # print(np.sum(target>0.5))
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

"""
Dice 系数
衡量两个样本集合的相似度，常用于医学图像分割等场景。
"""
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

"""
像素准确率
衡量预测正确的像素占总像素的比例。
"""
def pixel_acc(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    output_ = np.array(output > 0.5)
    target_ = np.array(target > 0.5)
    
    # FP = np.float(np.sum((output_==True) & (target_==False)))
    # FN = np.float(np.sum((output_==False) & (target_==True)))
    # TP = np.float(np.sum((output_==True) & (target_==True)))
    # TN = np.float(np.sum((output_==False) & (target_==False)))

    FP = float(np.sum((output_==True) & (target_==False)))
    FN = float(np.sum((output_==False) & (target_==True)))
    TP = float(np.sum((output_==True) & (target_==True)))
    TN = float(np.sum((output_==False) & (target_==False)))
    
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)

    return accuracy