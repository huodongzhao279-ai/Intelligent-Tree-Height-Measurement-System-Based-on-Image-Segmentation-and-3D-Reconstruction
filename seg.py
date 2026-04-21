import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

"""
通过用户指定的点来生成目标的掩码。
掩码生成函数:
    generate_mask() 接收图像路径和输入点，通过 SAM 生成目标掩码。
采用两次预测优化结果：
    第一次预测获取多个候选掩码及其分数。
    选择分数最高的掩码作为输入，进行第二次预测以得到更精确的结果。
关键参数:
    multimask_output=True：生成多个候选掩码，便于选择最优解。
    mask_input：使用第一次预测的最优掩码作为输入，引导模型优化分割边界。
"""

#允许程序在同一个进程中多次加载Intel Math Kernel Library (MKL) 的多个副本，避免程序崩溃
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 在代码开头添加

# # 定义show_mask函数（假设其为自定义函数）
# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

sam_type = "vit_b"
checkpoint_path = "models/sam/sam_vit_b_01ec64.pth"

# 强制使用 CUDA
if not torch.cuda.is_available():
    print("警告: SAM - CUDA 不可用，将使用 CPU")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"SAM 使用 CUDA: {torch.cuda.get_device_name(0)}")

# 模型是否已经初始化
_preloaded_sam = None

#初始化并返回模型实例
def get_sam_model():
    global _preloaded_sam, sam_type, checkpoint_path
    if _preloaded_sam is None:
        sam_model = sam_model_registry[sam_type](checkpoint=checkpoint_path)
        _preloaded_sam = sam_model.to(device)
    return _preloaded_sam

# 获取模型实例
sam = get_sam_model()


# # 初始化自动生成器
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,       # 每边采点数
#     pred_iou_thresh=0.88,     # 掩码质量阈值
#     stability_score_thresh=0.9,
#     crop_n_layers=1           # 多尺度分割层数
# )

# # 生成分割结果
# image = cv2.imread("inputs//data//images//1.png")
# # masks = mask_generator.generate(image)

def generate_mask(image_path, input_points):
    """
    生成黑白掩码。
    参数:
        image_path (str): 图片路径。
        input_point (list): 用户点击的坐标，格式为 [[x, y]]。
    返回: numpy.ndarray: 黑白掩码。
    """
    # 初始化预测器
    predictor = SamPredictor(sam)
    
    # 读取图片
    image = cv2.imread(image_path)
    predictor.set_image(image)
    
    # 设置输入点和标签
    # 设置两个正向提示点
    input_points = np.array(input_points)
    input_labels = np.array([1, 1])

    # 先执行预测获取 logits 和 scores
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True  # 获取多个候选结果
    )

    # 现在可以安全使用 logits 和 scores
    best_idx = np.argmax(scores)
    mask_input = logits[best_idx, :, :]

    # 二次预测优化结果
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        mask_input=mask_input[None, :, :],
        multimask_output=False
    )
    # # 预测掩码
    # masks, scores, logits = predictor.predict(
    #     point_coords=input_point,
    #     point_labels=input_label,
    #     mask_input=mask_input[None, :, :],
    #     multimask_output=False  # 只输出一个掩码
    # )
    # mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

    
    if masks is None or len(masks) == 0:
        raise ValueError("未生成任何掩码") 
    # # 获取第一个掩码
    # mask = masks[0]["segmentation"]
    mask = masks[0]  
    return mask.astype(np.uint8) * 255  # 转换为黑白二值图像
    # return mask