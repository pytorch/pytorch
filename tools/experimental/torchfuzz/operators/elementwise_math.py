# pyre-strict
"""Elementwise math Operator subclasses.

Covers unary, bool-output unary, binary, ternary, and a handful of special-case
elementwise ops (clamp_max, clamp_min, heaviside, ldexp, logcumsumexp). Op
classes that overlap upstream torchfuzz (add, sub, mul, div, clamp) are NOT
defined here — they remain registered by upstream.

Subclasses are generated via small factory helpers to avoid hundreds of lines
of near-identical boilerplate. Each generated class is assigned to a
module-level name and has ``__module__`` set so the introspection-based
registry discovers it.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    FLOAT_DTYPES,
    is_float_dtype,
    random_broadcast_shape,
)
from torchfuzz.operators.base import Operator
from torchfuzz.operators.tensor_pointwise import PointwiseOperatorBase
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# ---------------------------------------------------------------------------
# Base classes (excluded from introspection by the ``Base`` suffix)
# ---------------------------------------------------------------------------


class UnaryElementwiseOperatorBase(Operator):
    """Base for one-tensor-in / one-tensor-out elementwise math (same dtype)."""

    requires_float: bool = False

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        if self.requires_float and not is_float_dtype(output_spec.dtype):
            return False
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        return [
            TensorSpec(
                size=output_spec.size,  # pyrefly: ignore[missing-argument]
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        return f"{output_name} = {self.torch_op_name}({input_names[0]})"


class BoolUnaryElementwiseOperatorBase(Operator):
    """Base for one-numeric-tensor-in / one-bool-tensor-out (isfinite/isnan/...)."""

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype == torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        # Pick a random float dtype at spec-fuzz time
        return [
            TensorSpec(
                size=output_spec.size,  # pyrefly: ignore[missing-argument]
                stride=output_spec.stride,
                dtype=random.choice(
                    FLOAT_DTYPES + (torch.int8, torch.int16, torch.int32, torch.int64)
                ),
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        return f"{output_name} = {self.torch_op_name}({input_names[0]})"


class FloatOnlyPointwiseOperatorBase(PointwiseOperatorBase):
    """PointwiseOperatorBase restricted to float dtypes."""

    _scalar_input_positions = ()

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        return is_float_dtype(output_spec.dtype)


class TernaryElementwiseOperatorBase(Operator):
    """Base for three-tensor-in / one-tensor-out elementwise math (float-only)."""

    requires_float: bool = True

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        if self.requires_float and not is_float_dtype(output_spec.dtype):
            return False
        return True

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 2)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        return specs

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if len(input_names) != 3:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 3 input tensors"
            )
        return (
            f"{output_name} = {self.torch_op_name}("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]})"
        )


# ---------------------------------------------------------------------------
# Subclass factories
# ---------------------------------------------------------------------------


def _class_name_from_op(op_name: str) -> str:
    """Convert ``"torch.nan_to_num"`` -> ``"NanToNumOperator"``."""
    suffix = op_name.replace("torch.", "")
    return "".join(p.capitalize() for p in suffix.split("_")) + "Operator"


def _make_subclass(
    base: type[Operator],
    op_name: str,
    extra_attrs: dict[str, Any] | None = None,
) -> type[Operator]:
    """Build a concrete Operator subclass for ``op_name`` deriving from ``base``."""
    cls_name = _class_name_from_op(op_name)

    def _init(self: Operator) -> None:
        base.__init__(self, op_name)

    attrs: dict[str, Any] = {
        "__init__": _init,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    cls = type(cls_name, (base,), attrs)
    cls.__module__ = __name__
    return cls


# ---------------------------------------------------------------------------
# Unary subclasses
# ---------------------------------------------------------------------------

# Float-only unary ops (per plan filtering rules).
_UNARY_FLOAT_OPS: tuple[str, ...] = (
    "torch.acos",
    "torch.acosh",
    "torch.angle",
    "torch.arccos",
    "torch.arccosh",
    "torch.arcsin",
    "torch.arcsinh",
    "torch.arctan",
    "torch.arctanh",
    "torch.asin",
    "torch.asinh",
    "torch.atan",
    "torch.atanh",
    "torch.cos",
    "torch.cosh",
    "torch.deg2rad",
    "torch.erf",
    "torch.erfc",
    "torch.erfinv",
    "torch.exp",
    "torch.exp2",
    "torch.expm1",
    "torch.fix",
    "torch.frac",
    "torch.log",
    "torch.log10",
    "torch.log1p",
    "torch.log2",
    "torch.rad2deg",
    "torch.reciprocal",
    "torch.rsqrt",
    "torch.sin",
    "torch.sinh",
    "torch.sqrt",
    "torch.tan",
)

# Numeric unary ops that work on int + float (no bool).
_UNARY_NUMERIC_OPS: tuple[str, ...] = (
    "torch.abs",
    "torch.absolute",
    "torch.ceil",
    "torch.floor",
    "torch.neg",
    "torch.negative",
    "torch.real",
    "torch.sgn",
    "torch.sign",
    "torch.square",
    "torch.trunc",
)


for _op in _UNARY_FLOAT_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        UnaryElementwiseOperatorBase, _op, {"requires_float": True}
    )

for _op in _UNARY_NUMERIC_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        UnaryElementwiseOperatorBase, _op
    )


class LogitOperator(UnaryElementwiseOperatorBase):
    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.logit")
        self._eps: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._eps = round(random.uniform(1e-7, 1e-2), 8)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        eps = self._eps
        self._eps = None
        return f"{output_name} = torch.logit({input_names[0]}, eps={eps!r})"


class NanToNumOperator(UnaryElementwiseOperatorBase):
    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.nan_to_num")
        self._nan: float | None = None
        self._posinf: float | None = None
        self._neginf: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._nan = round(random.uniform(-10.0, 10.0), 4)
        self._posinf = round(random.uniform(0.0, 100.0), 4)
        self._neginf = round(random.uniform(-100.0, 0.0), 4)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        nan, posinf, neginf = self._nan, self._posinf, self._neginf
        self._nan = self._posinf = self._neginf = None
        return (
            f"{output_name} = torch.nan_to_num({input_names[0]}, "
            f"nan={nan!r}, posinf={posinf!r}, neginf={neginf!r})"
        )


class RoundOperator(UnaryElementwiseOperatorBase):
    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.round")
        self._decimals: int | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._decimals = random.randint(0, 4)
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        decimals = self._decimals
        self._decimals = None
        return f"{output_name} = torch.round({input_names[0]}, decimals={decimals})"


# ---------------------------------------------------------------------------
# Bool-output unary subclasses
# ---------------------------------------------------------------------------

_BOOL_UNARY_OPS: tuple[str, ...] = (
    "torch.isfinite",
    "torch.isinf",
    "torch.isnan",
    "torch.isneginf",
    "torch.isposinf",
    "torch.signbit",
)

for _op in _BOOL_UNARY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        BoolUnaryElementwiseOperatorBase, _op
    )


# ---------------------------------------------------------------------------
# Binary subclasses
# ---------------------------------------------------------------------------

# Float-only binary ops — both args must be tensors.
_BINARY_FLOAT_TENSOR_ONLY_OPS: tuple[str, ...] = (
    "torch.arctan2",
    "torch.atan2",
    "torch.hypot",
    "torch.logaddexp",
    "torch.logaddexp2",
    "torch.nextafter",
)

# Float-only binary ops — scalar allowed in position 1 (the `other` arg).
_BINARY_FLOAT_SCALAR_OTHER_OPS: tuple[str, ...] = ("torch.copysign",)

# Float-only binary ops — scalar allowed in either position.
_BINARY_FLOAT_SCALAR_BOTH_OPS: tuple[str, ...] = ("torch.xlogy",)

# Numeric binary ops — scalar allowed in either position.
_BINARY_NUMERIC_OPS: tuple[str, ...] = (
    "torch.floor_divide",
    "torch.multiply",
    "torch.remainder",
    "torch.subtract",
)

# Numeric binary ops — both args must be tensors.
_BINARY_NUMERIC_TENSOR_ONLY_OPS: tuple[str, ...] = (
    "torch.fmax",
    "torch.fmin",
    "torch.maximum",
    "torch.minimum",
)

# Numeric binary ops — scalar allowed in position 1 only.
_BINARY_NUMERIC_SCALAR_OTHER_OPS: tuple[str, ...] = ("torch.fmod",)

for _op in _BINARY_FLOAT_TENSOR_ONLY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op
    )

for _op in _BINARY_FLOAT_SCALAR_OTHER_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op, {"_scalar_input_positions": (1,)}
    )

for _op in _BINARY_FLOAT_SCALAR_BOTH_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        FloatOnlyPointwiseOperatorBase, _op, {"_scalar_input_positions": (0, 1)}
    )

for _op in _BINARY_NUMERIC_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(PointwiseOperatorBase, _op)

for _op in _BINARY_NUMERIC_TENSOR_ONLY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        PointwiseOperatorBase, _op, {"_scalar_input_positions": ()}
    )

for _op in _BINARY_NUMERIC_SCALAR_OTHER_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        PointwiseOperatorBase, _op, {"_scalar_input_positions": (1,)}
    )


class PowOperator(PointwiseOperatorBase):
    """torch.pow with non-negative exponents for integer bases.

    Negative integer exponents produce fractions that cannot be
    represented in integer dtypes; PyTorch raises RuntimeError.
    Uses Python abs() instead of torch.abs() so scalar exponents work.
    """

    def __init__(self) -> None:
        super().__init__("torch.pow")

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if (
            isinstance(output_spec, TensorSpec)
            and output_spec.dtype not in FLOAT_DTYPES
        ):
            return f"{output_name} = torch.pow({input_names[0]}, abs({input_names[1]}))"
        return super().codegen(output_name, input_names, output_spec)


class DivideOperator(PointwiseOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.divide")
        self._rounding_mode: str | None = None

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        self._rounding_mode = random.choice(["default", "trunc", "floor"])
        return super().fuzz_inputs_specs(output_spec, num_inputs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        rm = self._rounding_mode
        self._rounding_mode = None
        base = super().codegen(output_name, input_names, output_spec)
        if rm != "default":
            base = base[:-1] + f", rounding_mode={rm!r})"
        return base


class TrueDivideOperator(PointwiseOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.true_divide")


class RsubOperator(PointwiseOperatorBase):
    _scalar_input_positions = (1,)

    def __init__(self) -> None:
        super().__init__("torch.rsub")
        self._alpha: float | int | None = None

    def fuzz_inputs_specs(self, output_spec: Spec, num_inputs: int = 2) -> list[Spec]:
        if random.random() < 0.3:
            alpha = round(random.uniform(-5.0, 5.0), 4)
            if (
                isinstance(output_spec, TensorSpec)
                and output_spec.dtype not in FLOAT_DTYPES
            ):
                alpha = int(alpha)
            self._alpha = alpha
        else:
            self._alpha = None
        return super().fuzz_inputs_specs(output_spec, num_inputs)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        alpha = self._alpha
        self._alpha = None
        base = super().codegen(output_name, input_names, output_spec)
        if alpha is not None:
            base = base[:-1] + f", alpha={alpha!r})"
        return base


# ---------------------------------------------------------------------------
# Ternary subclasses (float-only)
# ---------------------------------------------------------------------------

_TERNARY_OPS: tuple[str, ...] = ("torch.lerp",)

for _op in _TERNARY_OPS:
    globals()[_class_name_from_op(_op)] = _make_subclass(
        TernaryElementwiseOperatorBase, _op
    )


class AddcdivOperator(TernaryElementwiseOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.addcdiv")
        self._value: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._value = (
            round(random.uniform(-5.0, 5.0), 4) if random.random() < 0.3 else None
        )
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        value = self._value
        self._value = None
        if len(input_names) != 3:
            raise ValueError("AddcdivOperator requires exactly 3 input tensors")
        base = (
            f"{output_name} = torch.addcdiv("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]}"
        )
        if value is not None:
            return f"{base}, value={value!r})"
        return f"{base})"


class AddcmulOperator(TernaryElementwiseOperatorBase):
    def __init__(self) -> None:
        super().__init__("torch.addcmul")
        self._value: float | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._value = (
            round(random.uniform(-5.0, 5.0), 4) if random.random() < 0.3 else None
        )
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        value = self._value
        self._value = None
        if len(input_names) != 3:
            raise ValueError("AddcmulOperator requires exactly 3 input tensors")
        base = (
            f"{output_name} = torch.addcmul("
            f"{input_names[0]}, {input_names[1]}, {input_names[2]}"
        )
        if value is not None:
            return f"{base}, value={value!r})"
        return f"{base})"


# ---------------------------------------------------------------------------
# Special-case operators
# ---------------------------------------------------------------------------


def _clamp_bound(output_spec: Spec) -> float | int:
    assert isinstance(output_spec, TensorSpec)  # noqa: S101
    bound = round(random.uniform(-10.0, 10.0), 4)
    if output_spec.dtype not in FLOAT_DTYPES:
        bound = int(bound)
    return bound


class ClampMaxOperator(Operator):
    """``torch.clamp_max(input, max)`` — random scalar bound."""

    def __init__(self) -> None:
        super().__init__("torch.clamp_max")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype != torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # Generate the bound here (not as instance state): the same operator
        # instance serves multiple clamp nodes, so stashing the bound in
        # fuzz_inputs_specs and reading it back here yielded max=None.
        bound = _clamp_bound(output_spec)
        return f"{output_name} = torch.clamp_max({input_names[0]}, max={bound!r})"


class ClampMinOperator(Operator):
    """``torch.clamp_min(input, min)`` — random scalar bound."""

    def __init__(self) -> None:
        super().__init__("torch.clamp_min")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype != torch.bool

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        bound = _clamp_bound(output_spec)
        return f"{output_name} = torch.clamp_min({input_names[0]}, min={bound!r})"


class HeavisideOperator(Operator):
    """``torch.heaviside(input, values)`` — second arg is a 0-d tensor."""

    def __init__(self) -> None:
        super().__init__("torch.heaviside")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype == torch.bool:
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("HeavisideOperator can only produce TensorSpec outputs")
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            ),
            TensorSpec(size=(), stride=(), dtype=output_spec.dtype),
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        if len(input_names) != 2:
            raise ValueError("HeavisideOperator requires exactly 2 input tensors")
        return f"{output_name} = torch.heaviside({input_names[0]}, {input_names[1]})"


class LdexpOperator(Operator):
    """``torch.ldexp(input, other)`` — second arg is an integer exponent."""

    _EXP_DTYPES: tuple[torch.dtype, ...] = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    def __init__(self) -> None:
        super().__init__("torch.ldexp")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        return is_float_dtype(output_spec.dtype)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            ),
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=random.choice(self._EXP_DTYPES),
            ),
        ]

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        return f"{output_name} = torch.ldexp({input_names[0]}, {input_names[1]})"
