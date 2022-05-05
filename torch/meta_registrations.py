import torch

Library = torch.library.Library
aten = torch.ops.aten

meta_lib = Library("aten", "IMPL", "Meta")

# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
def meta_index_select(*args, **kwargs):
    self, dim, index = args
    assert not kwargs
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)

def meta_inverse(*args, **kwargs):
    (self,) = args
    assert not kwargs
    if self.numel() == 0:
        return self.new_empty(self.size())
    inverse = self.new_empty(self.size())
    inverse.transpose_(-2, -1)
    return inverse

def meta_max(*args, **kwargs):
    (self,) = args
    assert not kwargs
    return self.new_empty(())

def meta_abs(*args, **kwargs):
    (self,) = args
    assert not kwargs
    if self.is_complex():
        from_complex = {torch.cfloat: torch.float, torch.cdouble: torch.double}
        float_type = from_complex[self.dtype]
        self.new_empty(self.size(), dtype=float_type)
    else:
        return self.new_empty(self.size())

meta_lib.impl(aten.index_select.default, meta_index_select)
meta_lib.impl(aten.inverse.default, meta_inverse)
meta_lib.impl(aten.max.default, meta_max)
meta_lib.impl(aten.abs.default, meta_abs)
