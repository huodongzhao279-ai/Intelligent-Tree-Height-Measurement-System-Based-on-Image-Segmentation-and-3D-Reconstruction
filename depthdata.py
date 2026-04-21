import numpy as np
import matplotlib.pyplot as plt
"""
实现了从 txt 文件加载深度数据、处理无效值，
并通过 matplotlib 实现交互式可视化（点击显示坐标和深度）的功能
"""


encodings_to_try = ['utf-8', 'gb18030', 'latin-1', 'iso-8859-1']

def load_depth_txt(file_path):
    """加载.txt深度数据文件"""

    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        row = [float(x) for x in line.split()]
                        data.append(row)
            print(f"成功使用编码: {enc}")
            return np.array(data)
            break
        except UnicodeDecodeError:
            continue    
       

def on_click(event):
    """鼠标点击事件回调函数"""
    if event.inaxes != ax:  # 确保点击在图像区域内
        return
    
    # 获取点击位置的像素坐标
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    
    # 获取实际深度值
    depth_value = depth_map_clean[y, x]  # 注意：y对应行，x对应列
    
    # 更新文本显示
    text.set_text(f'Position: ({x}, {y})\nDepth: {depth_value:.2f} m')
    plt.draw()

# 主程序
# 1. 加载数据并处理无效值
depth_map = load_depth_txt("input/1680935393752_depthdata.txt")
depth_map_clean = np.where(depth_map > 0, depth_map, np.nan)  # 无效值设为NaN

# 2. 计算深度范围
min_depth = np.nanmin(depth_map_clean)
max_depth = np.nanmax(depth_map_clean)

# 3. 创建可视化窗口
fig, ax = plt.subplots(figsize=(10, 6))

# 非归一化直接显示原始深度值
img = ax.imshow(
    depth_map_clean,
    cmap='jet',
    vmin=min_depth,  # 颜色映射范围=实际深度范围
    vmax=max_depth
)
plt.colorbar(img, label='Depth (m)')  # 颜色条显示实际单位
plt.title('ARCore Depth Map (Raw Values)')
plt.axis('off')

# 4. 添加交互文本
text = ax.text(
    0.05, 0.95,  # 文本位置（相对坐标）
    'Click anywhere to show depth',
    transform=ax.transAxes,
    color='white',
    backgroundcolor='black',
    verticalalignment='top'
)

# 5. 绑定鼠标点击事件
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()