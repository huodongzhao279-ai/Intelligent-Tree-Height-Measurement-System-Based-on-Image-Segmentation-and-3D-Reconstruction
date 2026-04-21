"""
定义了两个适用于深度学习任务（尤其是语义分割等二值化相关任务）的损失函数类，基于 PyTorch 框架实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

"""
结合二元交叉熵损失（BCE） 和Dice 损失的复合损失函数，常用于解决类别不平衡的二值分割任务。
"""
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

"""
基于lovasz_hinge损失函数的封装，
Lovasz 损失是针对分割任务设计的损失函数，对类别不平衡和边界区域的误差更敏感，能提升分割精度
"""
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
