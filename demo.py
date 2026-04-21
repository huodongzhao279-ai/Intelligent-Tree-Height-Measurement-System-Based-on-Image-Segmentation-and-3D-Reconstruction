from segment_anything import sam_model_registry, SamPredictor
import cv2  # 处理图像
import matplotlib.pyplot as plt  # 显示结果
import numpy as np

#解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为「黑体」（系统自带）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 2. 配置模型权重文件路径
model_path = "models/sam/sam_vit_b_01ec64.pth"  # 如果文件和代码同目录，直接写文件名即可

# 3. 加载模型
# "vit_b" 对应我们下载的权重（ViT-B 变体）
sam = sam_model_registry["vit_b"](checkpoint=model_path)
predictor = SamPredictor(sam)  # 创建“预测器”，后续用它做分割

# 4. 读取测试图片
image = cv2.imread("inputs/tree960/images/1.png")
# SAM 要求图片是 RGB 格式，而 OpenCV 读进来是 BGR，所以转个格式（固定写法）
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 5. 让模型“记住”这张图（预处理，固定步骤）
predictor.set_image(image)

# 6. 定义“提示点”
input_point = [[400, 400]]
# 提示点的“标签”：1 表示“这是要分割的物体”，0 表示“这不是”（固定写 1 就行）
input_label = [1]

# 将 list 转换为 numpy array
np_point_coords = np.array(input_point)
np_point_labels = np.array(input_label)

# 7. 执行分割（核心步骤，一行代码）
masks, scores, logits = predictor.predict(
    point_coords=np_point_coords,
    point_labels=np_point_labels,
    multimask_output=True,  # 是否返回多个可能的掩码（选 True，多一个选择）
)

# 8. 显示结果（不用懂，运行后会弹出窗口）
plt.figure(figsize=(10, 10))
# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(image)
# 在原图上画提示点（红色圆点，方便确认位置）
plt.scatter(input_point[0][0], input_point[0][1], color='red', s=100)
plt.title("原图 + 提示点")
plt.axis('off')

# 显示分割结果（选得分最高的掩码，scores[0] 是最高分）
plt.subplot(1, 2, 2)
plt.imshow(image)
# 叠加掩码（白色是分割出的物体，透明度 0.5）
plt.imshow(masks[0], alpha=0.5)
plt.title(f"分割结果（得分：{scores[0]:.2f}）")
plt.axis('off')

# 弹出窗口显示
plt.show()