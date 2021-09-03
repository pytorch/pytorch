# from torch.nn.modules import padding
import torch

# dtypes = (torch.uint8, torch.int8, torch.int16,
#           torch.int32, torch.int64, torch.float,
#           torch.double)

# for dtype in dtypes:

#     t = torch.randn(1, 4, 5, 5).to(dtype)
#     w = torch.randn(3, 4, 3, 3).to(dtype)

#     torch.nn.functional.conv2d(t, w)
#     try:
#         torch.nn.functional.conv2d(t, w, dilation=(2, 2))
#     except Exception as e:
#         print(dtype, ":", e)

def fn(t, w):
    return torch.nn.functional.conv2d(t, w)

print(torch.jit.script(fn).graph)