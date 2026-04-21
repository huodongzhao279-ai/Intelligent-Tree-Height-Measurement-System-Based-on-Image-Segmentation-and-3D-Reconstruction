# TreeHeight + SLAM3R 集成说明

## 概述

本项目将 **SLAM3R** (CVPR 2025 Highlight) 集成到树木高度测量系统中，实现了**无需手机深度传感器**即可进行三维重建和树木高度测量的功能。

### SLAM3R 论文信息
- **标题**: SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos
- **会议**: CVPR 2025 (Highlight Paper)
- **链接**: [arXiv](https://arxiv.org/abs/2412.09401)

## 系统对比

| 特性 | 原系统 (手机深度) | SLAM3R 模式 |
|------|------------------|-------------|
| 深度传感器 | ✅ 需要 | ❌ 不需要 |
| 多帧图像 | ❌ 单帧 | ✅ 支持多帧（效果更好） |
| 设备兼容性 | 仅支持带深度传感器的手机 | 任何相机/手机 |
| GPU 要求 | 可选 | 推荐 (CUDA) |

## 安装

### 1. 创建环境
```bash
conda create -n treeheight_slam3r python=3.11
conda activate treeheight_slam3r
```

### 2. 安装 PyTorch
```bash
# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch torchvision
```

### 3. 安装依赖
```bash
pip install -r requirements_slam3r.txt
```

### 4. (可选) 安装 xformers 加速
```bash
pip install xformers
```

## 使用方法

### 启动 SLAM3R 模式服务
```bash
python main_slam3r.py
```
服务将在 http://localhost:82 启动

### API 端点

#### 1. `/getData` (POST) - 兼容原 App
与原系统接口兼容，但**忽略深度数据**，仅使用图像进行重建。

#### 2. `/getDataSimple` (POST) - 简化接口
只需上传图像和坐标：
```bash
curl -X POST http://localhost:82/getDataSimple \
  -F "image=@tree.jpg" \
  -F "touchX1=364" \
  -F "touchY1=843" \
  -F "touchX2=758" \
  -F "touchY2=1216"
```

#### 3. `/get_point_cloud` (GET)
下载生成的点云文件。

## 文件结构

```
TreeHeight/
├── main.py                 # 原系统入口 (需要深度传感器)
├── main_slam3r.py          # SLAM3R 模式入口 (无需深度传感器)
├── slam3r_integration.py   # SLAM3R 集成模块
├── calculate_slam3r.py     # SLAM3R 计算模块
├── calculate.py            # 原计算模块
├── SLAM3R-main/            # SLAM3R 源代码
└── requirements_slam3r.txt # SLAM3R 依赖
```

## 工作流程

```
1. 用户上传 RGB 图像
        ↓
2. SAM 模型分割树木区域
        ↓
3. SLAM3R 从图像生成 3D 点云 (无需深度传感器!)
        ↓
4. 根据分割掩码提取树木点云
        ↓
5. 计算 AABB 包围盒获取树木高度
        ↓
6. 返回结果
```

## 注意事项

1. **首次运行**会自动从 HuggingFace 下载 SLAM3R 模型权重（约 2GB）
2. **GPU 推荐**：SLAM3R 在 GPU 上运行更快，CPU 模式较慢
3. **多帧效果更好**：如果能提供多张连续图像，三维重建效果会更好
4. **尺度校准**：SLAM3R 生成的点云可能需要根据实际场景进行尺度校准

## 测试

```bash
# 测试 SLAM3R 集成
python slam3r_integration.py

# 测试完整流程
python test_processing.py
```

## 常见问题

### Q: SLAM3R 模型下载失败？
A: 检查网络连接，或手动从 HuggingFace 下载模型放到缓存目录。

### Q: GPU 内存不足？
A: 尝试使用 CPU 模式，或减少图像分辨率。

### Q: 高度测量不准确？
A: SLAM3R 生成的点云尺度可能需要校准，可以在 `calculate_slam3r.py` 中调整缩放因子。
