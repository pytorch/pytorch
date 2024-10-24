

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


# A simple tensor subclass that holds a tensor with custom metadata and custom method
class ConstantExtraMetadataTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        shape = elem.shape
        kwargs = {}
        kwargs["strides"] = elem.stride()
        kwargs["storage_offset"] = elem.storage_offset()
        kwargs["device"] = elem.device
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        kwargs["dtype"] = elem.dtype
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, elem):
        self.elem = elem
        self.constant_attribute = 4

    def __repr__(self):
        inner_repr = repr(self.elem)
        return f"CustomTensor({inner_repr})"

    def __tensor_flatten__(self):
        return ["elem"], self.constant_attribute

    def add_constant(self, a):
        self.constant_attribute += a

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is not None
        elem = inner_tensors["elem"]
        out = ConstantExtraMetadataTensor(elem)
        out.constant_attribute = meta
        return out

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, args
        )

        kwargs_inner = pytree.tree_map_only(
            ConstantExtraMetadataTensor, lambda x: x.elem, kwargs
        )

        out_inner = func(*args_inner, **kwargs_inner)
        out_inner_flat, spec = pytree.tree_flatten(out_inner)
        # for aten ops that return non-tensors, just assume that
        # our cust inner tensors return the same value
        out_flat = [
            ConstantExtraMetadataTensor(o_inner)
            if isinstance(o_inner, torch.Tensor)
            else o_inner
            for o_inner in out_inner_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)
