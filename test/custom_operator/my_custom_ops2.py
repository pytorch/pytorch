import torch
from model import get_custom_op_library_path

torch.ops.load_library(get_custom_op_library_path())


@torch.library.impl_abstract("custom::sin")
def sin_abstract(x):
    return torch.empty_like(x)
