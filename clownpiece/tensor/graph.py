import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'TEST2.md')

# 设置中文字体
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
my_font = fm.FontProperties(fname=font_path)


# 读取文档数据
with open(file_path, 'r') as file:
    data = file.read()

# 提取测试尺寸和测试时间数据
patterns = {
    'Sequential': r'开始性能测试 \(Sequential\)(.*?)测试完成',
    'ThreadPerOp': r'开始性能测试 \(ThreadPerOp\)(.*?)测试完成',
    'OpenMP': r'开始性能测试 \(OpenMP\)(.*?)测试完成',
    'ThreadPool': r'开始性能测试 \(ThreadPool\)(.*?)测试完成',
}

element_wise_data = []
matrix_multiplication_data = []

for mode, pattern in patterns.items():
    match = re.search(pattern, data, re.DOTALL)
    if match:
        lines = match.group(1).strip().split('\n')
        for line in lines:
            if 'Element-wise' in line:
                size = int(re.search(r'(\d+)x(\d+)', line).group(1))
                time = float(re.search(r': (\d+\.\d+) ms', line).group(1))
                element_wise_data.append([mode, size, time])
            elif 'Matrix Multiplication' in line:
                size = int(re.search(r'(\d+)x(\d+)', line).group(1))
                time = float(re.search(r': (\d+\.\d+) ms', line).group(1))
                matrix_multiplication_data.append([mode, size, time])

# 转换为DataFrame
element_wise_df = pd.DataFrame(element_wise_data, columns=['Mode', 'Size', 'Time'])
matrix_multiplication_df = pd.DataFrame(matrix_multiplication_data, columns=['Mode', 'Size', 'Time'])

# 绘制逐元素操作的时间对比图
plt.figure(figsize=(10, 6))
for mode in element_wise_df['Mode'].unique():
    subset = element_wise_df[element_wise_df['Mode'] == mode]
    plt.plot(subset['Size'], subset['Time'], marker='o', label=mode)

plt.xlabel('测试尺寸', fontproperties=my_font)
plt.ylabel('测试时间 (ms)', fontproperties=my_font)
plt.title('逐元素操作的时间对比', fontproperties=my_font)
plt.legend()
plt.grid(True)
plt.show()

# 绘制矩阵乘法操作的时间对比图
plt.figure(figsize=(10, 6))
for mode in matrix_multiplication_df['Mode'].unique():
    subset = matrix_multiplication_df[matrix_multiplication_df['Mode'] == mode]
    plt.plot(subset['Size'], subset['Time'], marker='o', label=mode)

plt.xlabel('测试尺寸', fontproperties=my_font)
plt.ylabel('测试时间 (ms)', fontproperties=my_font)
plt.title('矩阵乘法操作的时间对比', fontproperties=my_font)
plt.legend()
plt.grid(True)
plt.show()