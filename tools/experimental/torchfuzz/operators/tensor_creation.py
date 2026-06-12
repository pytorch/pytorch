# pyre-strict
"""Tensor creation & initialization Operator subclasses.

These operators model torch factory functions (``torch.ones``, ``torch.full``,
``torch.arange``, ``torch.Tensor.new_full``, etc.) for the upstream torchfuzz
framework.  All factories are non-leaf operators: they may take zero or one
prototype tensor input but always produce a fresh ``TensorSpec`` output.

When ``output_spec.stride`` is non-contiguous, factories emit
``torch.empty_strided(...).fill_(...)`` (or a copy from a freshly-allocated
strided buffer for the ``_like`` family) so the requested layout is honored
without aliasing into another tensor's storage (see plan note [H19]).
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_ARANGE_ALLOWED_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }
)

_NON_BOOL_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    """Return the contiguous (row-major) stride for ``size``."""
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def _default_fill_value(dtype: torch.dtype) -> object:
    """Canonical fill value for full/full_like/new_full per dtype.

    Centralized so FullOperator, FullLikeOperator, and NewFullOperator cannot
    drift in their fill-value selection (per plan note [H4]).
    """
    if dtype == torch.bool:
        return True
    if dtype in FLOAT_DTYPES:
        return 1.0
    return 1


_UNSIGNED_INT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.uint8,
    }
)


def _random_fill_value(dtype: torch.dtype) -> object:
    """Random fill value appropriate for ``dtype``.

    Used by Full/FullLike/NewFull operators to exercise a wider range of
    constant values than the fixed ``_default_fill_value``.
    """
    if dtype == torch.bool:
        return random.choice([True, False])
    if dtype in FLOAT_DTYPES:
        return round(random.uniform(-10.0, 10.0), 4)
    if dtype in _UNSIGNED_INT_DTYPES:
        return random.randint(0, 100)
    return random.randint(-100, 100)


def _dtype_str(dt: torch.dtype) -> str:
    """Mirror the formatting used by upstream operators/constant.py."""
    return str(dt)


def _is_contiguous(output_spec: TensorSpec) -> bool:
    return tuple(output_spec.stride) == _contiguous_stride(tuple(output_spec.size))


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class TensorCreationFromSizeOperatorBase(Operator):
    """Base for factories that take no input tensor (ones, zeros, full)."""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        return []

    def _fill_value(self, output_spec: TensorSpec) -> object:
        """Subclasses override to provide the constant the factory writes."""
        raise NotImplementedError

    def _factory_args(self, output_spec: TensorSpec) -> str:
        """Args to pass after size for the contiguous-path factory call."""
        return f"dtype={_dtype_str(output_spec.dtype)}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        size = tuple(output_spec.size)
        if _is_contiguous(output_spec):
            return (
                f"{output_name} = {self.torch_op_name}({size}, "
                f"{self._factory_args(output_spec)})"
            )
        stride = tuple(output_spec.stride)
        return (
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}).fill_("
            f"{self._fill_value(output_spec)!r})"
        )


class TensorCreationFromPrototypeOperatorBase(Operator):
    """Base for factories that take one prototype tensor (ones_like, etc.)."""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        proto_dtype = output_spec.dtype
        # 30% chance: use a different prototype dtype to exercise the
        # dtype-override path (codegen passes dtype= explicitly).
        if proto_dtype != torch.bool and random.random() < 0.3:
            candidates = [dt for dt in _NON_BOOL_DTYPES if dt != proto_dtype]
            if candidates:
                proto_dtype = random.choice(candidates)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=proto_dtype,
            )
        ]

    def _like_extra_args(self, output_spec: TensorSpec) -> str:
        """Extra positional args after the prototype (e.g. fill value for full_like)."""
        return ""

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        prototype = input_names[0]
        extra = self._like_extra_args(output_spec)
        dtype_kwarg = f", dtype={_dtype_str(output_spec.dtype)}"
        like_call = f"{self.torch_op_name}({prototype}{extra}{dtype_kwarg})"
        if _is_contiguous(output_spec):
            return f"{output_name} = {like_call}"
        size = tuple(output_spec.size)
        stride = tuple(output_spec.stride)
        tmp = f"_tmp_{output_name}"
        return (
            f"{tmp} = {like_call}; "
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}); "
            f"{output_name}.copy_({tmp})"
        )


class TensorMethodNewOperatorBase(Operator):
    """Base for ``Tensor.new_X(size, ...)`` methods (new_ones, new_zeros, ...)."""

    method_name: str = ""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        return [TensorSpec(size=(1,), stride=(1,), dtype=output_spec.dtype)]

    def _fill_value(self, output_spec: TensorSpec) -> object:
        raise NotImplementedError

    def _new_extra_args(self, output_spec: TensorSpec) -> str:
        """Extra positional args after the size for the contiguous-path call."""
        return ""

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 prototype tensor"
            )
        prototype = input_names[0]
        size = tuple(output_spec.size)
        if _is_contiguous(output_spec):
            extra = self._new_extra_args(output_spec)
            return (
                f"{output_name} = {prototype}.{self.method_name}({size}{extra}, "
                f"dtype={_dtype_str(output_spec.dtype)})"
            )
        stride = tuple(output_spec.stride)
        return (
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}).fill_("
            f"{self._fill_value(output_spec)!r})"
        )


# ---------------------------------------------------------------------------
# Concrete factories without a prototype input
# ---------------------------------------------------------------------------


class OnesOperator(TensorCreationFromSizeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.ones")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return 1


class ZerosOperator(TensorCreationFromSizeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.zeros")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return 0


class FullOperator(TensorCreationFromSizeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.full")
        self._stashed_fill: object | None = None

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return self._stashed_fill

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return []

    def _factory_args(self, output_spec: TensorSpec) -> str:
        return f"{self._stashed_fill!r}, dtype={_dtype_str(output_spec.dtype)}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Concrete factories with a prototype input
# ---------------------------------------------------------------------------


class OnesLikeOperator(TensorCreationFromPrototypeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.ones_like")


class ZerosLikeOperator(TensorCreationFromPrototypeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.zeros_like")


class FullLikeOperator(TensorCreationFromPrototypeOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.full_like")
        self._stashed_fill: object | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return super().fuzz_inputs_specs(output_spec)

    def _like_extra_args(self, output_spec: TensorSpec) -> str:
        return f", {self._stashed_fill!r}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Concrete Tensor.new_X factories
# ---------------------------------------------------------------------------


class NewOnesOperator(TensorMethodNewOperatorBase):
    method_name: str = "new_ones"

    def __init__(self) -> None:
        super().__init__("torch.Tensor.new_ones")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return 1


class NewZerosOperator(TensorMethodNewOperatorBase):
    method_name: str = "new_zeros"

    def __init__(self) -> None:
        super().__init__("torch.Tensor.new_zeros")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return 0


class NewFullOperator(TensorMethodNewOperatorBase):
    method_name: str = "new_full"

    def __init__(self) -> None:
        super().__init__("torch.Tensor.new_full")
        self._stashed_fill: object | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return super().fuzz_inputs_specs(output_spec)

    def _fill_value(self, output_spec: TensorSpec) -> object:
        return self._stashed_fill

    def _new_extra_args(self, output_spec: TensorSpec) -> str:
        return f", {self._stashed_fill!r}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Special-case: torch.arange (always 1-D, contiguous)
# ---------------------------------------------------------------------------


class ArangeOperator(Operator):
    def __init__(self) -> None:
        super().__init__("torch.arange")
        self._start: int | None = None
        self._step: int | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if len(output_spec.size) != 1:
            return False
        if tuple(output_spec.stride) != (1,):
            return False
        return output_spec.dtype in _ARANGE_ALLOWED_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if random.random() < 0.5:
            self._start = random.randint(-10, 10)
            self._step = random.randint(1, 3)
        else:
            self._start = None
            self._step = None
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ArangeOperator can only produce TensorSpec outputs")
        n = output_spec.size[0]
        dtype_arg = f"dtype={_dtype_str(output_spec.dtype)}"
        if self._start is not None and self._step is not None:
            start = self._start
            step = self._step
            end = start + step * n
            result = (
                f"{output_name} = torch.arange({start}, {end}, {step}, {dtype_arg})"
            )
        else:
            result = f"{output_name} = torch.arange({n}, {dtype_arg})"
        self._start = None
        self._step = None
        return result
