"""Traceable data pointer primitive for Dynamo."""

from typing import Any

import torch
from torch import SymInt
from torch._prims import _make_prim, RETURN_TYPE
from torch._subclasses import FakeTensorMode


def _data_ptr_meta(self: torch.Tensor) -> int:
    return 0


def _data_ptr_impl_aten(self: torch.Tensor) -> int:
    return self.data_ptr()


_data_ptr = _make_prim(
    schema="_data_ptr(Tensor self) -> SymInt",
    return_type=RETURN_TYPE.NEW,
    meta=_data_ptr_meta,
    impl_aten=_data_ptr_impl_aten,
    doc="Traceable unbacked SymInt version of torch.Tensor.data_ptr().",
)


@_data_ptr.py_impl(FakeTensorMode)  # type: ignore[misc]
def _data_ptr_fake(fake_mode: FakeTensorMode, self_tensor: Any) -> SymInt:
    if fake_mode.shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")
    return fake_mode.shape_env.create_unbacked_symint()
