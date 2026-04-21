# 数据集格式

import os
import cv2
import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, point_dir, point_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            point_dir: 点坐标文件目录
            point_ext (str): 点坐标文件扩展名
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.point_dir = point_dir
        self.point_ext = point_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, str(img_id) + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, str(i), str(img_id) + self.mask_ext)
            class_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]  # 单通道掩码
            mask.append(class_mask)
        mask = np.dstack(mask) #拼接不同类别的mask，形状：(H, W, num_classes)

        # 加载点坐标
        point_path = os.path.join(self.point_dir, str(img_id) + self.point_ext)
        points = np.loadtxt(point_path, dtype=np.float32)  # 读取点坐标，格式为[x1 y1 x2 y2]
        points = points.reshape(2, 2)  # 转换为(2,2)形状，每个点包含(x,y)坐标

        # 数据增强（同步变换图像和掩码）
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 图像预处理（SAM默认输入为[0, 255]，但部分实现会归一化到[0,1]，此处保持与原逻辑一致）
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}, points
