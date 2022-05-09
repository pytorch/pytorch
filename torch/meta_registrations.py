import torch

Library = torch.library.Library

meta_lib = Library("aten", "IMPL", "Meta")

# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@torch.library.impl(meta_lib, "index_select")
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)

@torch.library.impl(meta_lib, "inverse")
def meta_inverse(self):
    if self.numel() == 0:
        return self.new_empty(self.size())
    inverse = self.new_empty(self.size())
    inverse.transpose_(-2, -1)
    return inverse

@torch.library.impl(meta_lib, "max")
def meta_max(self):
    return self.new_empty(())

@torch.library.impl(meta_lib, "abs")
def meta_abs(self):
    if self.is_complex():
        float_type = self.real.dtype
        self.new_empty(self.size(), dtype=float_type)
    else:
        return self.new_empty(self.size())
