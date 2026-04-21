"""
用于评估SAM (sam_vit_b_01ec64.pth) 模型性能的脚本
主要功能：加载SAM模型，在验证集上进行推理，计算IoU、像素准确率等指标并保存预测结果
"""

import argparse
import os
from glob import glob
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from segment_anything import sam_model_registry, SamPredictor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import Dataset
from metrics import iou_score, pixel_acc
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', default='sam_vit_b_01ec64.pth',
                        help='SAM模型权重路径')
    parser.add_argument('--dataset', default='tree50',
                        help='数据集名称')
    parser.add_argument('--img_ext', default='.png',
                        help='图像文件扩展名')
    parser.add_argument('--mask_ext', default='.png',
                        help='掩码文件扩展名')
    parser.add_argument('--point_ext', default='.txt',
                        help='点坐标文件的扩展名')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='类别数量（SAM默认输出二值掩码）')
    parser.add_argument('--device', default='cuda',
                        help='使用的设备（cuda或cpu）')
    return parser.parse_args()


def main():
    args = parse_args()
    cudnn.benchmark = True

    # 初始化SAM模型
    print("=> 加载SAM模型...")
    sam = sam_model_registry["vit_b"](checkpoint="models/sam/" + args.sam_checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    # 加载验证集数据
    img_ids = glob(os.path.join('inputs', args.dataset, 'images', '*' + args.img_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 划分训练集和验证集,使用90%训练10%验证的划分
    # _, val_img_ids = train_test_split(img_ids, test_size=0.1, random_state=41)
    val_img_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # 准备验证数据集
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', args.dataset, 'images'),
        mask_dir=os.path.join('inputs', args.dataset, 'masks'),
        img_ext=args.img_ext,
        mask_ext=args.mask_ext,
        point_dir=os.path.join('inputs', args.dataset, 'points'),
        point_ext=args.point_ext,
        num_classes=args.num_classes,
        transform=None  # SAM会自动处理图像预处理
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # SAM推理通常单张图像处理
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    # 初始化评估指标
    avg_meters = {
        'iou': AverageMeter(),
        'acc': AverageMeter()
    }

    # 创建输出目录
    for c in range(args.num_classes):
        os.makedirs(os.path.join('outputs', 'sam_vit_b', str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta, points in tqdm(val_loader, total=len(val_loader)):
            # 处理输入图像（SAM需要RGB格式和numpy数组）
            img_np = input.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            img_np = (img_np * 255).astype(np.uint8)  # 转换为0-255范围

            # 设置SAM输入图像
            predictor.set_image(img_np)

            # 处理目标掩码和点坐标
            target_np = target.squeeze().cpu().numpy().astype(np.uint8)
            points_np = points.squeeze().cpu().numpy()  # 提取点坐标，形状为(2,2)

            if np.sum(target_np) > 0:  # 确保存在目标
                # 计算目标掩码的边界框
                contours, _ = cv2.findContours(target_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                bbox = [x, y, x + w, y + h]  # [x1, y1, x2, y2]

                # 使用边界框提示生成预测掩码
                masks, _, _ = predictor.predict(
                    box=np.array(bbox),
                    point_coords=points_np,
                    point_labels=np.array([1, 1]),  # 标记两个点均为前景（1表示前景，0表示背景）
                    multimask_output=False  # 只输出一个最佳掩码
                )
                output = masks[0].astype(np.float32)  # 取第一个掩码
            else:
                # 无目标时输出全0掩码
                output = np.zeros_like(target_np, dtype=np.float32)

            # 转换为张量格式以便计算指标
            output_tensor = torch.from_numpy(output).unsqueeze(0).unsqueeze(0).to(args.device)
            target_tensor = target.to(args.device)

            # 计算评估指标
            iou = iou_score(output_tensor, target_tensor)
            acc = pixel_acc(output_tensor, target_tensor)
            avg_meters['iou'].update(iou, 1)
            avg_meters['acc'].update(acc, 1)

            # 保存预测结果
            for c in range(args.num_classes):
                save_path = os.path.join('outputs', 'sam_vit_b', str(c), f"{meta['img_id'][0]}.png")
                cv2.imwrite(save_path, (output * 255).astype('uint8'))

    # 打印最终评估结果
    print(f"IoU: {avg_meters['iou'].avg:.4f}, Pixel Accuracy: {avg_meters['acc'].avg:.4f}")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()