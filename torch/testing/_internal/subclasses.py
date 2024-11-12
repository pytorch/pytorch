# mypy: ignore-errors
from typing import Any, Optional, Type

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
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, a.size(), **kwargs)

        return out

    def __init__(self, a, outer_size=None, outer_stride=None):
        self.a = a

    def __repr__(self):
        return f"WrapperSubclass({repr(self.a)})"

    def __tensor_flatten__(self):
        return ["a"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a = inner_tensors["a"]
        if is_fake(a):
            assert outer_size is not None
            assert outer_stride is not None
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
        self, expected_metadata: Any, expected_type: Optional[Type] = None
    ):
        if expected_type == type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None


def quant_rw(src, scale, bias, dtype):  # scale, bias are tensors
    assert scale.ndim == 1
    assert bias.ndim == 1
    assert src.size(0) == scale.size(0)
    assert src.size(0) == bias.size(0)
    if src.ndim == 1:
        b = bias
        s = scale
    else:
        b = bias.expand(src.size(1), src.size(0)).t()
        s = scale.expand(src.size(1), src.size(0)).t()
    sub = src - b
    return (sub / s).to(dtype)


class QuantRWTensorBase(torch.Tensor):
    DTYPE = torch.int32
    QDTYPE = torch.int8

    @staticmethod
    def __new__(cls, qdata, scale, bias):
        assert qdata.dtype == cls.QDTYPE
        src = qdata
        shape = src.shape
        kwargs = {}
        kwargs["strides"] = src.stride()
        kwargs["storage_offset"] = src.storage_offset()
        kwargs["device"] = src.device
        kwargs["layout"] = src.layout
        kwargs["requires_grad"] = src.requires_grad
        kwargs["dtype"] = cls.DTYPE
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, qdata, scale, bias):
        assert qdata.dtype == self.QDTYPE
        assert qdata.size(0) == scale.size(0)
        assert qdata.size(0) == bias.size(0)
        assert scale.dtype == self.DTYPE
        assert bias.dtype == self.DTYPE
        assert scale.ndim == 1
        assert bias.ndim == 1
        self.qdata = qdata
        self.scale = scale
        self.bias = bias

    @classmethod
    def from_src(cls, src):
        if isinstance(src, cls):
            return src

        scale = torch.ones((src.size(0),), dtype=cls.DTYPE)
        bias = torch.zeros((src.size(0),), dtype=cls.DTYPE)
        qdata = quant_rw(src, scale, bias, cls.QDTYPE)
        return cls(qdata, scale, bias)

    def dequant(self):
        if self.qdata.ndim == 1:
            b = self.bias
            s = self.scale
        else:
            assert self.qdata.ndim == 2
            b = self.bias.expand(self.qdata.size(1), self.qdata.size(0)).t()
            s = self.scale.expand(self.qdata.size(1), self.qdata.size(0)).t()
        return (self.qdata * s + b).to(self.DTYPE)

    def __repr__(self):
        return f"{self.__class__.__name__}.from_src({self.dequant()})"

    def __tensor_flatten__(self):
        return ["qdata", "scale", "bias"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        qdata, scale, bias = (
            inner_tensors["qdata"],
            inner_tensors["scale"],
            inner_tensors["bias"],
        )
        return cls(qdata, scale, bias)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_dequant = pytree.tree_map_only(cls, lambda x: x.dequant(), args)
        kwargs_dequant = pytree.tree_map_only(cls, lambda x: x.dequant(), kwargs)

        raw_out = func(*args_dequant, **kwargs_dequant)
        out_flat, spec = pytree.tree_flatten(raw_out)
        res_out_flat = [
            (
                cls.from_src(o)
                if isinstance(o, torch.Tensor) and o.dtype == cls.DTYPE
                else o
            )
            for o in out_flat
        ]
        out = pytree.tree_unflatten(res_out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out

        return return_and_correct_aliasing(func, args, kwargs, out)


class I32QuantRWTensor(QuantRWTensorBase):
    DTYPE = torch.int16
    QDTYPE = torch.int8


class F32_QI32QuantRWTensor(QuantRWTensorBase):
    DTYPE = torch.float32
    QDTYPE = torch.int16

    @classmethod
    def from_src(cls, src):
        scale = torch.full((src.size(0),), 1, dtype=cls.DTYPE)
        bias = torch.full((src.size(0),), 1, dtype=cls.DTYPE)
        qdata = quant_rw(src, scale, bias, dtype=cls.QDTYPE)
        qqdata = I32QuantRWTensor.from_src(qdata)
        return cls(qqdata, scale, bias)
