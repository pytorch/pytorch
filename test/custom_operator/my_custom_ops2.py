from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


@torch.library.register_fake("custom::sin")
def sin_abstract(x):
    return torch.empty_like(x)
