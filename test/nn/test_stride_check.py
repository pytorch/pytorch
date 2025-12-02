import torch
import torch.nn.quantized as nnq

try:
    # 错误输入
    conv_layer = nnq.Conv2d(3, 16, kernel_size=3, stride=0, padding=1)
except RuntimeError as e:
    print("捕获错误:", e)

# 正确输入
conv_layer = nnq.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
print("卷积创建成功:", conv_layer)
