from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


# NB: The impl_abstract_pystub for cos actually
# specifies it should live in the my_custom_ops2 module.
@torch.library.register_fake("custom::cos")
def cos_abstract(x):
    return torch.empty_like(x)


# NB: There is no impl_abstract_pystub for tan
@torch.library.register_fake("custom::tan")
def tan_abstract(x):
    return torch.empty_like(x)
