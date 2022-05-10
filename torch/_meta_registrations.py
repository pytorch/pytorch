import torch

meta_lib = torch.library.Library("aten", "IMPL", "Meta")

# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@torch.library.impl(meta_lib, "index_select")
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)

@torch.library.impl(meta_lib, "index_select.out")
def meta_index_select_out(self, dim, index, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.index_select(self, dim, index))

@torch.library.impl(meta_lib, "abs")
def meta_abs(self):
    if self.is_complex():
        from_complex = {torch.complex32: torch.half, torch.cfloat: torch.float, torch.cdouble: torch.double}
        float_type = from_complex[self.dtype]
        return self.new_empty(self.size(), dtype=float_type)
    else:
        return self.new_empty(self.size())

@torch.library.impl(meta_lib, "abs.out")
def meta_abs_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.abs(self))

@torch.library.impl(meta_lib, "max")
def meta_max(self):
    return self.new_empty(())

@torch.library.impl(meta_lib, "min")
def meta_min(self):
    return self.new_empty(())
