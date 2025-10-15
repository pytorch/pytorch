"""This module implements tensor version operations for Dynamo tracing.

It provides primitives for handling tensor versioning during tracing, particularly in the
context of functionalization where version operations are handled eagerly on fake tensors.

When we functionalize _tensor_version + _unsafe_set_version_counter, the ops disappear from
the traced graph. We run them eagerly on the fake tensors used for tracing, in order to get
past asserts that would fail in autograd.

Why is this ok?
1) Versions on functional tensors do not make any sense since you cannot mutate a functional
   tensor.
2) The whole point of version munging is to trick autograd into doing what we want, and after
   AotAutograd there is no longer any need for these ops.

Note this is similar to how no_grad is handled.
"""

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import SymInt
from torch._prims import _make_prim, RETURN_TYPE
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode


_tensor_version = _make_prim(
    schema="_tensor_version(Tensor self) -> SymInt",
    return_type=RETURN_TYPE.NEW,
    meta=torch.ops.aten._version.default,
    impl_aten=torch.ops.aten._version.default,
    doc="Tracable unbacked SymInt version of torch.Tensor._version",
)


@_tensor_version.py_impl(FakeTensorMode)  # type: ignore[misc]
def _tensor_version_fake(fake_mode: FakeTensorMode, self_tensor: Any) -> SymInt:
    """
    The initial dynamo capture of _tensor_version + _unsafe_set_version_counter turns the
    `._version` into an unbacked SymInt so that we don't need to specialize on the `._version`
    of input tensors to the graph.
    """
    assert fake_mode.shape_env is not None
    return fake_mode.shape_env.create_unbacked_symint()


_unsafe_set_version_counter = _make_prim(
    schema="_unsafe_set_version_counter(Tensor[] tensors, SymInt[] versions) -> ()",
    return_type=RETURN_TYPE.NEW,
    meta=lambda self, version: None,
    impl_aten=torch._C._autograd._unsafe_set_version_counter,
    doc="Tracable+SymInt version of torch._C._autograd._unsafe_set_version_counter",
)
torch.fx.node.has_side_effect(_unsafe_set_version_counter)


@_tensor_version.py_impl(FunctionalTensorMode)  # type: ignore[misc]
def _tensor_version_functional(mode: FunctionalTensorMode, self: Any) -> int:
    return self._version


@_unsafe_set_version_counter.py_impl(FunctionalTensorMode)  # type: ignore[misc]
def _unsafe_set_version_counter_functional(
    ctx: AbstractContextManager[Any],
    tensors: tuple[torch.Tensor, ...],
    versions: tuple[int, ...],
) -> None:
    torch._C._autograd._unsafe_set_version_counter(tensors, versions)
