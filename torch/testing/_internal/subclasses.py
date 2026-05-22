# mypy: ignore-errors
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import is_fake
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


class WrapperSubclass(torch.Tensor):
    @staticmethod
    def __new__(cls, a, outer_size=None, outer_stride=None):
        if outer_size is None:
            outer_size = a.size()
        if outer_stride is None:
            outer_stride = a.stride()

        kwargs = {}
        kwargs["strides"] = outer_stride
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, outer_size, **kwargs)

        return out

    def __init__(self, a, outer_size=None, outer_stride=None):
        self.a = a

    def __repr__(self):
        return f"WrapperSubclass({repr(self.a)})"

    def __tensor_flatten__(self):
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        if meta is not None:
            raise AssertionError("Expected meta to be None")
        a = inner_tensors["a"]
        if is_fake(a):
            if outer_size is None:
                raise AssertionError("Expected outer_size to not be None")
            if outer_stride is None:
                raise AssertionError("Expected outer_stride to not be None")
        return WrapperSubclass(a, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, args)

        kwargs_a = pytree.tree_map_only(WrapperSubclass, lambda x: x.a, kwargs)

        out_a = func(*args_a, **kwargs_a)
        out_a_flat, spec = pytree.tree_flatten(out_a)
        out_flat = [
            WrapperSubclass(o_a) if isinstance(o_a, torch.Tensor) else o_a
            for o_a in out_a_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: type | None = None
    ):
        if expected_type is type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None
