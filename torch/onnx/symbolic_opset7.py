import torch
import torch.onnx
import torch.onnx.utils

from torch.onnx.symbolic_helper import _black_list_in_opset
from torch.onnx.symbolic_opset9 import _adaptive_pool, _pair, _single, _triple, max_pool1d, max_pool2d, max_pool3d

black_listed_operators = ["scan", "expand"]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)

adaptive_max_pool1d = _adaptive_pool('adaptive_max_pool1d', "MaxPool", _single, max_pool1d)
adaptive_max_pool2d = _adaptive_pool('adaptive_max_pool2d', "MaxPool", _pair, max_pool2d)
adaptive_max_pool3d = _adaptive_pool('adaptive_max_pool3d', "MaxPool", _triple, max_pool3d)