import torch
import torch._custom_ops as library
from model import get_custom_op_library_path

torch.ops.load_library(get_custom_op_library_path())


@library.impl_abstract("custom::sin")
def sin_abstract(x):
    return x.sin()
