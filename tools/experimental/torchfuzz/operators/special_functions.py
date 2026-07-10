# pyre-strict
"""Special-function Operator subclasses.

Models torch ops in the "Special Functions" category. All 12 ops
are float-only and produce a TensorSpec output of the same
shape/stride/dtype as their primary input (the `x` argument).

Three sub-groupings:

  * 9 unary float-only ops -- thin subclasses of
    UnaryElementwiseOperatorBase from elementwise_math.py with
    `requires_float = True`. The base's `codegen` template works
    for `torch.special.X` ops because Python evaluates
    `torch.special.bessel_j0` as a single attribute-lookup
    expression. (digamma, i0, lgamma, sinc, bessel_j0, bessel_j1,
    bessel_y0, bessel_y1, i0e)

  * IgammaOperator -- direct Operator subclass that emits
    `torch.clamp` on both inputs in codegen to satisfy
    `torch.igamma`'s domain constraints (a > 0, x >= 0).

  * ChebyshevPolynomialTOperator / ChebyshevPolynomialUOperator --
    share ChebyshevPolynomialOperatorBase. Pick a random small
    non-negative int n (0..5) at fuzz_inputs_specs time.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import (
    contiguous_stride,
    FLOAT_DTYPES,
    random_broadcast_shape,
)
from torchfuzz.operators.base import Operator
from torchfuzz.operators.elementwise_math import UnaryElementwiseOperatorBase
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


# Per-dtype clamp min for the `a` argument of torch.igamma. 1e-6
# underflows to 0 in bfloat16, defeating the a > 0 constraint;
# bf16 uses 1e-2 instead.
_IGAMMA_A_MIN: dict[torch.dtype, float] = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-6,
    torch.float32: 1e-6,
    torch.float64: 1e-6,
}


# ---------------------------------------------------------------------------
# Group A: 9 unary float-only ops
# ---------------------------------------------------------------------------


class DigammaOperator(UnaryElementwiseOperatorBase):
    """torch.digamma(x) -- logarithmic derivative of the gamma function.

    Wraps input in ``torch.abs(x) + 1e-6`` to keep values positive and
    away from the poles at non-positive integers.
    """

    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.digamma")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        return f"{output_name} = torch.digamma(torch.abs({input_names[0]}) + 1e-6)"


class I0Operator(UnaryElementwiseOperatorBase):
    """torch.i0(x) -- modified Bessel function of the first kind, order 0."""

    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.i0")

    @property
    def torch_op_name(self) -> str:
        return self.name


class LgammaOperator(UnaryElementwiseOperatorBase):
    """torch.lgamma(x) -- natural log of the absolute value of the gamma function.

    Wraps input in ``torch.abs(x) + 1e-6`` to keep values positive and
    away from the poles at non-positive integers.
    """

    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.lgamma")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        return f"{output_name} = torch.lgamma(torch.abs({input_names[0]}) + 1e-6)"


class SincOperator(UnaryElementwiseOperatorBase):
    """torch.sinc(x) -- normalized sinc function: sin(pi*x) / (pi*x)."""

    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.sinc")

    @property
    def torch_op_name(self) -> str:
        return self.name


class _BesselOperatorBase(UnaryElementwiseOperatorBase):
    """Shared base for bessel_j0/j1/y0/y1 -- CPU kernels only support fp32/fp64."""

    requires_float = True

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in (torch.float32, torch.float64)


class BesselJ0Operator(_BesselOperatorBase):
    """torch.special.bessel_j0(x) -- Bessel function of the first kind, order 0."""

    def __init__(self) -> None:
        super().__init__("torch.special.bessel_j0")

    @property
    def torch_op_name(self) -> str:
        return self.name


class BesselJ1Operator(_BesselOperatorBase):
    """torch.special.bessel_j1(x) -- Bessel function of the first kind, order 1."""

    def __init__(self) -> None:
        super().__init__("torch.special.bessel_j1")

    @property
    def torch_op_name(self) -> str:
        return self.name


class BesselY0Operator(_BesselOperatorBase):
    """torch.special.bessel_y0(x) -- Bessel function of the second kind, order 0.

    Wraps input in ``torch.abs(x) + 1e-6`` because bessel_y0 returns
    -inf at x=0 and NaN for x<0.
    """

    def __init__(self) -> None:
        super().__init__("torch.special.bessel_y0")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        return (
            f"{output_name} = torch.special.bessel_y0("
            f"torch.abs({input_names[0]}) + 1e-6)"
        )


class BesselY1Operator(_BesselOperatorBase):
    """torch.special.bessel_y1(x) -- Bessel function of the second kind, order 1.

    Wraps input in ``torch.abs(x) + 1e-6`` because bessel_y1 returns
    -inf at x=0 and NaN for x<0.
    """

    def __init__(self) -> None:
        super().__init__("torch.special.bessel_y1")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        return (
            f"{output_name} = torch.special.bessel_y1("
            f"torch.abs({input_names[0]}) + 1e-6)"
        )


class I0eOperator(UnaryElementwiseOperatorBase):
    """torch.special.i0e(x) -- exponentially scaled modified Bessel function I0e."""

    requires_float = True

    def __init__(self) -> None:
        super().__init__("torch.special.i0e")

    @property
    def torch_op_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Group B: IgammaOperator
# ---------------------------------------------------------------------------


class IgammaOperator(Operator):
    """torch.igamma(a, x) -- regularized lower incomplete gamma function.

    Requires a > 0 and x >= 0. Inputs are clamped at codegen time:
      - a is clamped with a per-dtype `min` (1e-2 for bfloat16,
        1e-6 for all other float dtypes).
      - x is clamped to min=0.0 for all float dtypes.

    Both inputs are float tensors matching the output spec. No
    per-call statefulness; the clamp bounds are fixed-per-dtype
    constants.
    """

    def __init__(self) -> None:
        super().__init__("torch.igamma")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return isinstance(output_spec, TensorSpec) and output_spec.dtype in FLOAT_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        spec = TensorSpec(
            size=output_spec.size,
            stride=output_spec.stride,
            dtype=output_spec.dtype,
        )
        specs: list[Spec] = [spec, spec]
        if random.random() < 0.3:
            idx = random.randint(0, 1)
            bcast_size = random_broadcast_shape(tuple(output_spec.size))
            if bcast_size != tuple(output_spec.size):
                specs[idx] = TensorSpec(
                    size=bcast_size,
                    stride=contiguous_stride(bcast_size),
                    dtype=output_spec.dtype,
                )
        return specs

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        a_min = _IGAMMA_A_MIN[output_spec.dtype]
        return (
            f"{output_name} = torch.igamma("
            f"torch.clamp({input_names[0]}, min={a_min}), "
            f"torch.clamp({input_names[1]}, min=0.0))"
        )


# ---------------------------------------------------------------------------
# Group C: Chebyshev polynomial base + 2 subclasses
# ---------------------------------------------------------------------------


class ChebyshevPolynomialOperatorBase(Operator):
    """Shared base for chebyshev_polynomial_t and chebyshev_polynomial_u.

    Both ops have signature ``op(x, n)`` where x is a float tensor and
    n is the non-negative integer polynomial degree. n is picked at
    fuzz_inputs_specs time via random.randint(0, 5), stashed on
    self._n, and emitted inline at codegen time as
    `torch.tensor(n, dtype=torch.int32)`. Only x is a fuzzed input
    tensor.

    The class name suffix `Base` is excluded from the introspection-
    based registration.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._n: int | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        return output_spec.dtype in (torch.float32, torch.float64)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._n = random.randint(0, 5)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # is not None -- NOT truthiness -- n=0 is a valid draw.
        try:
            return (
                f"{output_name} = {self.torch_op_name}("
                f"{input_names[0]}, "
                f"torch.tensor({self._n}, dtype=torch.int32))"
            )
        finally:
            self._n = None


class ChebyshevPolynomialTOperator(ChebyshevPolynomialOperatorBase):
    """torch.special.chebyshev_polynomial_t(x, n) -- Chebyshev T_n(x)."""

    def __init__(self) -> None:
        super().__init__("torch.special.chebyshev_polynomial_t")

    @property
    def torch_op_name(self) -> str:
        return self.name


class ChebyshevPolynomialUOperator(ChebyshevPolynomialOperatorBase):
    """torch.special.chebyshev_polynomial_u(x, n) -- Chebyshev U_n(x)."""

    def __init__(self) -> None:
        super().__init__("torch.special.chebyshev_polynomial_u")

    @property
    def torch_op_name(self) -> str:
        return self.name
