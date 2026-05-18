"""Python access to ATen's TensorIterator build pipeline.

Two equivalent ways to build an iterator are exposed:

1. The fluent ``TensorIteratorConfig`` (see :class:`TensorIteratorConfig`) with
   chained ``.add_input(...)`` / ``.add_output(...)`` / setter calls, mirroring
   the C++ ``at::TensorIteratorConfig`` API.
2. A dataclass-style ``ConfigSpec`` that carries the same knobs as fields.

Both ultimately drive the same C++ builder. Inspection of the built iterator
goes through :class:`TensorIterator`, which wraps ``torch._C._TensorIterator``
and exposes shape/dtype/device/stride information.

This is a build-pipeline-only surface: there is no ``for_each`` here. Use it
to debug shape and dtype inference, validate custom-op contracts, or inspect
how ATen would lay out a kernel's iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch._C import (
    _TensorIterator as _CTensorIterator,
    _TensorIteratorConfig as _CTensorIteratorConfig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "TensorIterator",
    "TensorIteratorConfig",
    "ConfigSpec",
    "binary_op",
    "binary_float_op",
    "comparison_op",
    "unary_op",
    "unary_float_op",
    "nullary_op",
    "reduce_op",
]


class TensorIterator:
    """A built TensorIterator. Read-only view of the build-pipeline result.

    Construct via :meth:`TensorIteratorConfig.build`, the dataclass-style
    :func:`build_from_spec`, or one of the factory shortcuts.
    """

    def __init__(self, impl: _CTensorIterator) -> None:
        self._impl = impl

    @property
    def ndim(self) -> int:
        return self._impl.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._impl.shape

    @property
    def numel(self) -> int:
        return self._impl.numel

    @property
    def ntensors(self) -> int:
        return self._impl.ntensors

    @property
    def ninputs(self) -> int:
        return self._impl.ninputs

    @property
    def noutputs(self) -> int:
        return self._impl.noutputs

    @property
    def is_contiguous(self) -> bool:
        return self._impl.is_contiguous

    @property
    def is_trivial_1d(self) -> bool:
        return self._impl.is_trivial_1d

    @property
    def common_dtype(self) -> torch.dtype | None:
        """The promoted computation dtype, or ``None`` if promotion was not requested."""
        return self._impl.common_dtype

    def tensor(self, index: int) -> torch.Tensor:
        return self._impl.tensor(index)

    def input(self, index: int = 0) -> torch.Tensor:
        return self._impl.input(index)

    def output(self, index: int = 0) -> torch.Tensor:
        return self._impl.output(index)

    def dtype(self, index: int = 0) -> torch.dtype:
        return self._impl.dtype(index)

    def device(self, index: int = 0) -> torch.device:
        return self._impl.device(index)

    def strides(self, index: int) -> tuple[int, ...]:
        """Per-operand strides in bytes (post reorder/coalesce)."""
        return self._impl.strides(index)

    def element_strides(self, index: int) -> tuple[int, ...]:
        """Per-operand strides in elements (byte stride / element size)."""
        return self._impl.element_strides(index)

    def __repr__(self) -> str:
        return repr(self._impl)


class TensorIteratorConfig:
    """Fluent builder that mirrors ``at::TensorIteratorConfig``.

    Outputs must be added before any inputs (matching the C++ contract);
    calling :meth:`add_output` after :meth:`add_input` raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._impl = _CTensorIteratorConfig()

    def add_output(self, tensor: torch.Tensor | None) -> TensorIteratorConfig:
        """Register an output. Pass ``None`` to ask the iterator to allocate."""
        self._impl.add_output(tensor)
        return self

    def add_input(self, tensor: torch.Tensor) -> TensorIteratorConfig:
        self._impl.add_input(tensor)
        return self

    def add_const_input(self, tensor: torch.Tensor) -> TensorIteratorConfig:
        self._impl.add_const_input(tensor)
        return self

    def check_all_same_dtype(self, value: bool) -> TensorIteratorConfig:
        self._impl.check_all_same_dtype(value)
        return self

    def check_all_same_device(self, value: bool) -> TensorIteratorConfig:
        self._impl.check_all_same_device(value)
        return self

    def promote_inputs_to_common_dtype(self, value: bool) -> TensorIteratorConfig:
        self._impl.promote_inputs_to_common_dtype(value)
        return self

    def promote_integer_inputs_to_float(self, value: bool) -> TensorIteratorConfig:
        self._impl.promote_integer_inputs_to_float(value)
        return self

    def cast_common_dtype_to_outputs(self, value: bool) -> TensorIteratorConfig:
        self._impl.cast_common_dtype_to_outputs(value)
        return self

    def enforce_safe_casting_to_output(self, value: bool) -> TensorIteratorConfig:
        self._impl.enforce_safe_casting_to_output(value)
        return self

    def enforce_linear_iteration(self, value: bool = True) -> TensorIteratorConfig:
        self._impl.enforce_linear_iteration(value)
        return self

    def resize_outputs(self, value: bool) -> TensorIteratorConfig:
        self._impl.resize_outputs(value)
        return self

    def set_check_mem_overlap(self, value: bool) -> TensorIteratorConfig:
        self._impl.set_check_mem_overlap(value)
        return self

    def allow_cpu_scalars(self, value: bool) -> TensorIteratorConfig:
        self._impl.allow_cpu_scalars(value)
        return self

    def is_reduction(self, value: bool) -> TensorIteratorConfig:
        self._impl.is_reduction(value)
        return self

    def declare_static_dtype(self, dtype: torch.dtype) -> TensorIteratorConfig:
        self._impl.declare_static_dtype(dtype)
        return self

    def declare_static_device(self, device: torch.device) -> TensorIteratorConfig:
        self._impl.declare_static_device(device)
        return self

    def declare_static_dtype_and_device(
        self, dtype: torch.dtype, device: torch.device
    ) -> TensorIteratorConfig:
        self._impl.declare_static_dtype_and_device(dtype, device)
        return self

    def declare_static_shape(
        self,
        shape: Sequence[int],
        squash_dims: Sequence[int] = (),
    ) -> TensorIteratorConfig:
        self._impl.declare_static_shape(list(shape), list(squash_dims))
        return self

    def build(self) -> TensorIterator:
        return TensorIterator(self._impl.build())


@dataclass
class ConfigSpec:
    """Dataclass-style alternative to :class:`TensorIteratorConfig`.

    Defaults match the C++ ``TensorIteratorConfig`` defaults. ``outputs`` are
    always registered before ``inputs`` and ``const_inputs``.
    """

    outputs: list[torch.Tensor | None] = field(default_factory=list)
    inputs: list[torch.Tensor] = field(default_factory=list)
    const_inputs: list[torch.Tensor] = field(default_factory=list)

    check_all_same_dtype: bool = True
    check_all_same_device: bool = True
    promote_inputs_to_common_dtype: bool = False
    promote_integer_inputs_to_float: bool = False
    cast_common_dtype_to_outputs: bool = False
    enforce_safe_casting_to_output: bool = False
    enforce_linear_iteration: bool = False
    resize_outputs: bool = True
    check_mem_overlap: bool = True
    allow_cpu_scalars: bool = False
    is_reduction: bool = False

    static_dtype: torch.dtype | None = None
    static_device: torch.device | None = None
    static_shape: Sequence[int] | None = None
    squash_dims: Sequence[int] = ()

    def build(self) -> TensorIterator:
        cfg = TensorIteratorConfig()
        # Outputs MUST come first.
        for t in self.outputs:
            cfg.add_output(t)
        for t in self.inputs:
            cfg.add_input(t)
        for t in self.const_inputs:
            cfg.add_const_input(t)

        # Apply the boolean knobs only when they diverge from the C++ defaults
        # so we mirror exactly what a C++ caller would have produced.
        if not self.check_all_same_dtype:
            cfg.check_all_same_dtype(False)
        if not self.check_all_same_device:
            cfg.check_all_same_device(False)
        if self.promote_inputs_to_common_dtype:
            cfg.promote_inputs_to_common_dtype(True)
        if self.promote_integer_inputs_to_float:
            cfg.promote_integer_inputs_to_float(True)
        if self.cast_common_dtype_to_outputs:
            cfg.cast_common_dtype_to_outputs(True)
        if self.enforce_safe_casting_to_output:
            cfg.enforce_safe_casting_to_output(True)
        if self.enforce_linear_iteration:
            cfg.enforce_linear_iteration(True)
        if not self.resize_outputs:
            cfg.resize_outputs(False)
        if not self.check_mem_overlap:
            cfg.set_check_mem_overlap(False)
        if self.allow_cpu_scalars:
            cfg.allow_cpu_scalars(True)
        if self.is_reduction:
            cfg.is_reduction(True)

        if self.static_dtype is not None and self.static_device is not None:
            cfg.declare_static_dtype_and_device(self.static_dtype, self.static_device)
        elif self.static_dtype is not None:
            cfg.declare_static_dtype(self.static_dtype)
        elif self.static_device is not None:
            cfg.declare_static_device(self.static_device)

        if self.static_shape is not None:
            cfg.declare_static_shape(self.static_shape, self.squash_dims)

        return cfg.build()


# --- Factory shortcuts. These mirror the C++ named constructors at
# aten/src/ATen/TensorIterator.cpp:1069+. Pass ``out=None`` to ask the iterator
# to allocate a fresh output tensor of the inferred shape/dtype/device.


def binary_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::binary_op``."""
    return (
        TensorIteratorConfig()
        .add_output(out)
        .add_const_input(a)
        .add_const_input(b)
        .set_check_mem_overlap(True)
        .allow_cpu_scalars(True)
        .promote_inputs_to_common_dtype(True)
        .cast_common_dtype_to_outputs(True)
        .enforce_safe_casting_to_output(True)
        .build()
    )


def binary_float_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::binary_float_op``."""
    return (
        TensorIteratorConfig()
        .add_output(out)
        .add_const_input(a)
        .add_const_input(b)
        .set_check_mem_overlap(True)
        .allow_cpu_scalars(True)
        .promote_inputs_to_common_dtype(True)
        .cast_common_dtype_to_outputs(True)
        .enforce_safe_casting_to_output(True)
        .promote_integer_inputs_to_float(True)
        .build()
    )


def comparison_op(
    out: torch.Tensor | None, a: torch.Tensor, b: torch.Tensor
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::comparison_op``.

    When ``out`` is ``None``, the output dtype is forced to bool. When ``out``
    is a defined non-bool tensor, the common dtype is cast back to its dtype
    via ``cast_common_dtype_to_outputs``. The bool-output case skips that cast
    as a performance optimization.
    """
    cfg = TensorIteratorConfig()
    cfg.set_check_mem_overlap(True)
    cfg.allow_cpu_scalars(True)
    cfg.promote_inputs_to_common_dtype(True)
    if out is None:
        cfg.declare_static_dtype(torch.bool)
    elif out.dtype != torch.bool:
        cfg.cast_common_dtype_to_outputs(True)
    return cfg.add_output(out).add_const_input(a).add_const_input(b).build()


def unary_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::unary_op``."""
    return (
        TensorIteratorConfig()
        .add_output(out)
        .add_const_input(a)
        .set_check_mem_overlap(True)
        .cast_common_dtype_to_outputs(False)
        .enforce_safe_casting_to_output(False)
        .check_all_same_dtype(True)
        .build()
    )


def unary_float_op(out: torch.Tensor | None, a: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::unary_float_op``."""
    return (
        TensorIteratorConfig()
        .add_output(out)
        .add_const_input(a)
        .set_check_mem_overlap(True)
        .promote_inputs_to_common_dtype(True)
        .cast_common_dtype_to_outputs(True)
        .enforce_safe_casting_to_output(True)
        .promote_integer_inputs_to_float(True)
        .build()
    )


def nullary_op(out: torch.Tensor) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::nullary_op``.

    Unlike the binary/unary factories, ``out`` must be a defined tensor;
    the C++ named constructor takes a non-undefined output.
    """
    if out is None:
        raise TypeError(
            "nullary_op requires a defined output tensor; None is not allowed."
        )
    return (
        TensorIteratorConfig()
        .set_check_mem_overlap(True)
        .check_all_same_dtype(False)
        .resize_outputs(False)
        .add_output(out)
        .build()
    )


def reduce_op(
    out: torch.Tensor,
    a: torch.Tensor,
    *,
    out2: torch.Tensor | None = None,
) -> TensorIterator:
    """Equivalent of ``at::TensorIterator::reduce_op``.

    Pass ``out2`` for the two-output reduction overload (e.g. ``min`` returning
    values + indices). The output tensor(s) must be pre-allocated and shaped
    correctly: this factory does not allocate or resize.
    """
    cfg = TensorIteratorConfig().set_check_mem_overlap(False).add_output(out)
    if out2 is not None:
        cfg.add_output(out2)
        return (
            cfg.add_const_input(a)
            .resize_outputs(False)
            .is_reduction(True)
            .check_all_same_dtype(False)
            .build()
        )
    return (
        cfg.add_const_input(a)
        .resize_outputs(False)
        .is_reduction(True)
        .promote_inputs_to_common_dtype(True)
        .build()
    )
