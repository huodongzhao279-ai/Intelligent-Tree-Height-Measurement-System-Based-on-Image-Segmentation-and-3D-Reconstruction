# -*- coding:UTF-8 -*-
"""
测试 CUDA 配置
"""
import torch

print("=" * 50)
print("CUDA 配置测试")
print("=" * 50)

# PyTorch 版本
print(f"\nPyTorch 版本: {torch.__version__}")

# CUDA 可用性
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # CUDA 版本
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # GPU 信息
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    # GPU 内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU 总内存: {total_memory:.2f} GB")
    
    # 测试 CUDA 计算
    print("\n测试 CUDA 计算...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"矩阵乘法测试: {z.shape} ✓")
    
    # cuDNN
    print(f"\ncuDNN 可用: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    
    print("\n" + "=" * 50)
    print("CUDA 配置正常，可以使用 GPU 加速！")
    print("=" * 50)
else:
    print("\n警告: CUDA 不可用")
    print("可能的原因:")
    print("  1. 没有 NVIDIA GPU")
    print("  2. 没有安装 CUDA 驱动")
    print("  3. PyTorch 安装的是 CPU 版本")
    print("\n请安装 GPU 版本的 PyTorch:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
