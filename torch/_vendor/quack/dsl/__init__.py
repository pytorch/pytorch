# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

"""CuTe DSL helpers and integration hooks."""

import torch._vendor.quack.dsl.cute_tensor_indexing  # noqa: F401
import torch._vendor.quack.dsl.cute_tensor  # noqa: F401
import torch._vendor.quack.dsl.mixed_constexpr_if  # noqa: F401
from torch._vendor.quack.dsl.torch_library_op import cute_op

__all__ = ["cute_op"]


def __getattr__(name: str):
    if name == "cute_op":
        from torch._vendor.quack.dsl.torch_library_op import cute_op

        return cute_op
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
