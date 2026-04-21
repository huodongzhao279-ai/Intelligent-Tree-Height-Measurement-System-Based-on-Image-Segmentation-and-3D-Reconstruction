# -*- coding:UTF-8 -*-
"""
测试 SLAM3R 集成
"""
import os
import sys

print("=" * 60)
print("SLAM3R 集成测试")
print("=" * 60)

# 测试1: 检查 SLAM3R 路径
print("\n[1] 检查 SLAM3R 路径...")
SLAM3R_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SLAM3R-main')
if os.path.exists(SLAM3R_PATH):
    print(f"    ✓ SLAM3R 路径存在: {SLAM3R_PATH}")
else:
    print(f"    ✗ SLAM3R 路径不存在: {SLAM3R_PATH}")
    sys.exit(1)

# 测试2: 检查 SLAM3R 模块导入
print("\n[2] 检查 SLAM3R 模块导入...")
sys.path.insert(0, SLAM3R_PATH)

try:
    from slam3r.models import Image2PointsModel, Local2WorldModel
    print("    ✓ SLAM3R 模型模块导入成功")
except ImportError as e:
    print(f"    ✗ SLAM3R 模型模块导入失败: {e}")
    print("    请先安装 SLAM3R 依赖: pip install -r requirements_slam3r.txt")
    sys.exit(1)

try:
    from slam3r.utils.image import load_images, ImgNorm
    print("    ✓ SLAM3R 工具模块导入成功")
except ImportError as e:
    print(f"    ✗ SLAM3R 工具模块导入失败: {e}")
    sys.exit(1)

# 测试3: 检查 PyTorch
print("\n[3] 检查 PyTorch...")
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"    PyTorch 版本: {torch.__version__}")
print(f"    CUDA 可用: {torch.cuda.is_available()}")
print(f"    使用设备: {device}")

# 测试4: 检查集成模块
print("\n[4] 检查集成模块...")
try:
    import slam3r_integration
    print("    ✓ slam3r_integration 模块导入成功")
    
    if slam3r_integration.check_slam3r_available():
        print("    ✓ SLAM3R 可用")
    else:
        print("    ✗ SLAM3R 不可用")
except ImportError as e:
    print(f"    ✗ slam3r_integration 导入失败: {e}")
    sys.exit(1)

# 测试5: 初始化模型 (可选，需要下载权重)
print("\n[5] 初始化 SLAM3R 模型...")
print("    注意: 首次运行会从 HuggingFace 下载模型权重 (~2GB)")
print("    如果网络较慢，可能需要等待较长时间")

user_input = input("    是否初始化模型? (y/n): ")
if user_input.lower() == 'y':
    try:
        success = slam3r_integration.init_slam3r_models(device)
        if success:
            print("    ✓ SLAM3R 模型初始化成功!")
        else:
            print("    ✗ SLAM3R 模型初始化失败")
    except Exception as e:
        print(f"    ✗ 初始化错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("    跳过模型初始化")

# 测试6: 测试图像预处理
print("\n[6] 测试图像预处理...")
test_images = [
    "uploads/input/1765801316099.jpg",
    "uploads/input/1765798513157.jpg"
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"    找到测试图像: {img_path}")
        try:
            view = slam3r_integration.preprocess_image(img_path)
            print(f"    ✓ 预处理成功: img shape={view['img'].shape}")
        except Exception as e:
            print(f"    ✗ 预处理失败: {e}")
        break
else:
    print("    未找到测试图像，跳过预处理测试")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)

print("""
下一步:
1. 如果所有测试通过，运行 SLAM3R 模式:
   python main_slam3r.py

2. 服务启动后，访问: http://localhost:82

3. 使用 /getDataSimple 接口测试:
   curl -X POST http://localhost:82/getDataSimple \\
     -F "image=@tree.jpg" \\
     -F "touchX1=364" -F "touchY1=843" \\
     -F "touchX2=758" -F "touchY2=1216"
""")
