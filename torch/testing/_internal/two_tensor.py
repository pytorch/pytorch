# mypy: ignore-errors

import torch
import torch.utils._pytree as pytree
from torch._subclasses.base import BaseTensorSubclass


# A simple tensor subclass that holds two tensors internally, and runs every op on both tensors.
class TwoTensor(BaseTensorSubclass):
    TSC_INNER_TENSORS = ["a", "b"]

    @staticmethod
    def __new__(cls, a, b, meta, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        assert (
            a.device == b.device
            and a.layout == b.layout
            and a.requires_grad == b.requires_grad
            and a.dtype == b.dtype
        )
        # I guess it would be more accurate to represent the shape as torch.cat(a, b).shape
        shape = outer_size
        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
        return out

    def __init__(self, a, b, meta, outer_size=None, outer_stride=None):
        self.a = a
        self.b = b
        self.meta = meta

    def get_meta(self):
        return (self.meta,)

    @staticmethod
    def meta_init_kwargs(meta):
        return {"meta": meta[0]}

    # @classmethod
    # def torch_function_prologue(cls, func, types, args=(), kwargs=None):
    #    print(f"torch_fn_prologue {cls} {func}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        out_a = cls.func_args_kwargs_attr(func, args, kwargs, "a")
        out_b = cls.func_args_kwargs_attr(func, args, kwargs, "b")

        _meta = None

        def fn(x):
            nonlocal _meta
            if not _meta:
                _meta = x.meta

        pytree.tree_map_only(cls, fn, args)
        if not _meta:
            pytree.tree_map_only(cls, fn, kwargs)

        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_b_flat = pytree.tree_leaves(out_b)
        # for aten ops that return non-tensors, just assume that
        # our two inner tensors return the same value
        out_flat = [
            cls(o_a, o_b, _meta) if isinstance(o_a, torch.Tensor) else o_a
            for o_a, o_b in zip(out_a_flat, out_b_flat)
        ]
        out = pytree.tree_unflatten(out_flat, spec)

        return cls._return(func, args, kwargs, out)


class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        return out
