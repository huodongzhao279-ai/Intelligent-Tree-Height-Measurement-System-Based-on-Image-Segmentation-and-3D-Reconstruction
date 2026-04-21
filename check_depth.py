import numpy as np
import os

input_dir = 'uploads/input'
depth_files = [f for f in os.listdir(input_dir) if f.endswith('_depthdata.txt')]

print("检查所有深度数据文件:")
for f in depth_files:
    path = os.path.join(input_dir, f)
    data = np.fromfile(path, dtype=np.uint16)
    print(f"  {f}: size={data.size}, min={data.min()}, max={data.max()}")
