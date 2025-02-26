# mypy: ignore-errors
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch._subclasses.base import (
    BaseTensorSubclass as BaseTSC,
    tensor_kwargs_from,
    torch_dispatch_override,
    torch_function_override,
)
from torch.testing._internal.two_tensor import TwoTensor


class WrapperSubclass(BaseTSC):
    TSC_INNER_TENSORS = ["a"]

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

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        out = pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(x),
            cls.func_args_kwargs_attr(func, args, kwargs or {}, "a"),
        )
        return cls._return(func, args, kwargs, out)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: Optional[type] = None
    ):
        if expected_type == type(self.a):
            return self.a
        elif expected_type is TwoTensor:
            return TwoTensor(self.a, self.a.clone())

        return None


class LogTensor(BaseTSC):
    TSC_INNER_TENSORS = ["a"]

    @staticmethod
    def __new__(
        cls,
        a: torch.Tensor,
        outer_size=None,
        outer_stride=None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls, outer_size or a.size(), **tensor_kwargs_from(a, outer_stride)
        )

    def __init__(
        self,
        a: torch.Tensor,
        outer_size=None,
        outer_stride=None,
    ):
        self.a = a

    @classmethod
    def torch_function_prologue(cls, func, types, args, kwargs) -> Optional[Any]:
        print(f"{cls} torch_function_prologue {func}")
        return None

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        print(f"{cls} torch_dispatch {func}")
        out = pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(x),
            cls.func_args_kwargs_attr(func, args, kwargs or {}, "a"),
        )
        return cls._return(func, args, kwargs, out)


class BaseWithMeta(BaseTSC):
    TSC_INNER_TENSORS = ["a"]
    TSC_META = ["m"]

    @staticmethod
    def __new__(
        cls,
        a: torch.Tensor,
        m: str,
        outer_size=None,
        outer_stride=None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls, outer_size or a.size(), **tensor_kwargs_from(a, outer_stride)
        )

    def __init__(
        self,
        a: torch.Tensor,
        m: str,
        outer_size=None,
        outer_stride=None,
    ):
        self.a = a
        self.m = m

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        ms = []

        def add_m(sc):
            ms.append(sc.m)

        pytree.tree_map_only(cls, add_m, args)
        pytree.tree_map_only(cls, add_m, kwargs)

        m = ms[0] if ms else "no_m"

        out = pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(x, m),
            cls.func_args_kwargs_attr(func, args, kwargs or {}, "a"),
        )
        return cls._return(func, args, kwargs, out)


class BaseWithOverride(BaseTSC):
    TSC_INNER_TENSORS = ["a"]

    @staticmethod
    def __new__(
        cls,
        a: torch.Tensor,
        outer_size=None,
        outer_stride=None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls, outer_size or a.size(), **tensor_kwargs_from(a, outer_stride)
        )

    def __init__(
        self,
        a: torch.Tensor,
        outer_size=None,
        outer_stride=None,
    ):
        self.a = a

    @torch_function_override(ops={torch.add})
    def torch_fn_add(cls, func, types, args=(), kwargs=None):  # noqa: B902
        print(f"{cls}.torch_fn_add {func} {types}")
        return func(*args, **kwargs)

    @staticmethod
    def tsc_unwrap_to_tensor(sc):
        return sc.a

    @classmethod
    def tsc_wrap_tensor(cls, t):
        return cls(t)

    @torch_dispatch_override(
        ops={torch.ops.aten.add.Tensor},
    )
    def torch_disp_add(cls, func, types, args=(), kwargs=None):  # noqa: B902
        # Calling func(args, kwargs) without unwrapping subclasses results in recursive cycle
        print(f"{cls}.torch_disp_add {func} {types}")
        return BaseTSC.default_torch_dispatch(cls, func, types, args, kwargs)
