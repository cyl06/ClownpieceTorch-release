import torch
import torch.nn as nn

# 创建输入张量 [批次大小, 通道数, 高度, 宽度]
input_tensor = torch.randn(1, 1, 3, 4)
print(f"输入形状: {input_tensor.shape}")

# 计算same padding所需的填充量
kernel_size = (2, 3)  # 卷积核高度和宽度
stride = (1, 1)      # 步长
dilation = (1, 1)    # 膨胀率

# 计算same padding的公式
padding_h = ((stride[0] * (3 - 1)) - 3 + dilation[0] * (kernel_size[0] - 1) + 1) // 2
padding_w = ((stride[1] * (4 - 1)) - 4 + dilation[1] * (kernel_size[1] - 1) + 1) // 2

# 创建卷积层
conv = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=kernel_size,
    stride=stride,
    padding='same',  # 手动计算的same padding值
    dilation=dilation
)

# 应用卷积
output = conv(input_tensor)
print(f"输出形状: {output.shape}")  # 应该保持与输入相同的空间尺寸 [3, 4]

# 打印输入和输出张量
print("\n输入张量:")
print(input_tensor.squeeze())
print("\n输出张量:")
print(output.squeeze().detach())    