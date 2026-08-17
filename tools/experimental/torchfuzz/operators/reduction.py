# pyre-strict
"""Reduction Operator subclasses.

Models torch ops in the "Reduction Ops" category. Five operator
families:

  A) Cumulative single-tensor (cumprod) -- same shape as input,
     pick random dim. Mirrors upstream CumsumOperator.
  B) Cumulative tuple .values (cummax, cummin) -- same as A but
     codegen extracts .values.
  C) Full reduction, single tensor (sum, mean, prod, nansum, var,
     all, any, amax, amin, argmax, argmin) -- produces either a 0-D
     scalar (no-dim form) or a keepdim=True-shaped output (one
     reduced dim of size 1).
  D) Dim-required tuple .values (max, min, median, mode) -- always
     uses the 2-arg form with keepdim=True, codegen extracts .values.
  E) Standalone special-case ops:
       - CountNonzeroOperator (0-D int64 output, no keepdim support)
       - AminmaxOperator (0-D output, .min extraction)
       - VarMeanOperator (0-D float output, [0] extraction)

torch.cumsum reuses upstream's CumsumOperator and has no class in
this file.

All operators that stash per-call random choices follow the
defense-in-depth pattern: __init__ initialises stash to None,
fuzz_inputs_specs populates from the global random module,
codegen defensively asserts ``is not None`` (NOT truthiness -- falsy
values are valid stash contents), then snapshots-and-clears the
stash to local variables BEFORE building the output string.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES, is_float_dtype
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_NUMERIC_INPUT_DTYPES_NO_BOOL: tuple[torch.dtype, ...] = (
    *FLOAT_DTYPES,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)

_BOOL_OR_NUMERIC_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.bool,
    *_NUMERIC_INPUT_DTYPES_NO_BOOL,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    """Standard row-major contiguous strides for ``size``."""
    if not size:
        return ()
    strides = [1]
    for s in reversed(size[1:]):
        strides.append(strides[-1] * s)
    return tuple(reversed(strides))


def _random_input_shape() -> tuple[int, ...]:
    """Pick a non-empty shape for full-reduction scalar inputs.

    ndim in [1, 4]; each dim in [1, 6]. Uses the global random module so
    --seed reproduces.
    """
    ndim = random.randint(1, 4)
    return tuple(random.randint(1, 6) for _ in range(ndim))


def _random_input_shape_min_numel_2() -> tuple[int, ...]:
    """Pick a non-empty input shape with total numel >= 2.

    Used by Bessel-corrected reductions (``torch.var``, ``torch.var_mean``)
    that divide by ``n - 1`` and return NaN at numel == 1, which would
    pollute numerics comparisons with spurious "NaN != NaN" mismatches.
    Bumps a random dim if every dim drew 1.
    """
    ndim = random.randint(1, 4)
    size = [random.randint(1, 6) for _ in range(ndim)]
    if all(s == 1 for s in size):
        idx = random.randrange(ndim)
        size[idx] = random.randint(2, 6)
    return tuple(size)


# ---------------------------------------------------------------------------
# Family A -- Cumulative single-tensor
# ---------------------------------------------------------------------------


class _CumulativeReductionBase(Operator):
    """Single-tensor cumulative reduction with same shape as input.

    Picks a random dim at fuzz_inputs_specs time and emits
    ``torch.X(input, dim=k)``.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._dim: int | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and len(output_spec.size) >= 1
            and output_spec.dtype in _NUMERIC_INPUT_DTYPES_NO_BOOL
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._dim = random.randrange(len(output_spec.size))
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
        dim = self._dim
        self._dim = None
        return f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim})"


class CumprodOperator(_CumulativeReductionBase):
    def __init__(self) -> None:
        super().__init__("torch.cumprod")

    @property
    def torch_op_name(self) -> str:
        return self.name


class LogcumsumexpOperator(_CumulativeReductionBase):
    """Float-only cumulative reduction: ``torch.logcumsumexp(input, dim)``."""

    def __init__(self) -> None:
        super().__init__("torch.logcumsumexp")

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and len(output_spec.size) >= 1
            and output_spec.dtype in FLOAT_DTYPES
        )


# ---------------------------------------------------------------------------
# Family B -- Cumulative tuple .values
# ---------------------------------------------------------------------------


class _CumulativeTupleReductionBase(_CumulativeReductionBase):
    """Same as _CumulativeReductionBase but codegen extracts .values."""

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        dim = self._dim
        self._dim = None
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim}).values"
        )


class CummaxOperator(_CumulativeTupleReductionBase):
    def __init__(self) -> None:
        super().__init__("torch.cummax")

    @property
    def torch_op_name(self) -> str:
        return self.name


class CumminOperator(_CumulativeTupleReductionBase):
    def __init__(self) -> None:
        super().__init__("torch.cummin")

    @property
    def torch_op_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Family C -- Full reduction, single tensor
# ---------------------------------------------------------------------------


class _FullOrKeepdimReductionBase(Operator):
    """Full reduction producing a 0-D scalar, keepdim=True, or keepdim=False shape.

    Subclasses set:
      - allowed_output_dtypes: tuple of torch.dtype that this op can
        produce as output.
      - input_dtype_pool: tuple of torch.dtype that the input may use
        when the input dtype is decoupled from the output dtype.
        When None (default), the input dtype equals the output dtype.
      - codegen_dtype_kwarg: when True (sum/prod/nansum), append
        ", dtype={output_spec.dtype}" to the emitted call.
      - supports_keepdim_false: when True, ``can_produce`` accepts
        any non-scalar shape (not just those with a size-1 dim).

    Modes:
      "scalar"  : output_spec.size == (); emit ``torch.X(input)``.
      "keepdim" : dim-specific reduction; emit
                  ``torch.X(input, dim=k, keepdim=True)`` or
                  ``torch.X(input, dim=k)`` (keepdim=False).
    """

    allowed_output_dtypes: tuple[torch.dtype, ...] = ()
    input_dtype_pool: tuple[torch.dtype, ...] | None = None
    codegen_dtype_kwarg: bool = False
    supports_keepdim_false: bool = False

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._mode: str | None = None
        self._dim: int | None = None
        self._keepdim: bool | None = None
        self._input_size: tuple[int, ...] | None = None
        self._input_dtype: torch.dtype | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype not in self.allowed_output_dtypes:
            return False
        if output_spec.size == ():
            return True
        if 1 in output_spec.size:
            return True
        return self.supports_keepdim_false

    def _pick_input_dtype(self, output_dtype: torch.dtype) -> torch.dtype:
        if self.input_dtype_pool is None:
            return output_dtype
        return random.choice(self.input_dtype_pool)

    def _pick_scalar_input_shape(self) -> tuple[int, ...]:
        """Hook for subclasses to override the scalar-mode input-shape draw.

        Default delegates to ``_random_input_shape``. Bessel-corrected
        reductions (e.g., ``torch.var``) override this to delegate to
        ``_random_input_shape_min_numel_2``.
        """
        return _random_input_shape()

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._input_dtype = self._pick_input_dtype(output_spec.dtype)
        if output_spec.size == ():
            self._mode = "scalar"
            self._keepdim = None
            self._input_size = self._pick_scalar_input_shape()
        elif 1 in output_spec.size and (
            not self.supports_keepdim_false or random.random() < 0.5
        ):
            self._mode = "keepdim"
            self._keepdim = True
            size_one_dims = [i for i, s in enumerate(output_spec.size) if s == 1]
            self._dim = random.choice(size_one_dims)
            new_dim = random.randint(2, 6)
            self._input_size = tuple(
                new_dim if i == self._dim else s for i, s in enumerate(output_spec.size)
            )
        else:
            self._mode = "keepdim"
            self._keepdim = False
            ndim = len(output_spec.size)
            self._dim = random.randint(0, ndim)
            r = random.randint(2, 6)
            input_size = list(output_spec.size)
            input_size.insert(self._dim, r)
            self._input_size = tuple(input_size)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=self._input_dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        mode = self._mode
        dim = self._dim
        keepdim = self._keepdim
        self._mode = None
        self._dim = None
        self._keepdim = None
        self._input_size = None
        self._input_dtype = None
        dtype_kwarg = f", dtype={output_spec.dtype}" if self.codegen_dtype_kwarg else ""
        if mode == "scalar":
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}{dtype_kwarg})"
            )
        if keepdim:
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}, "
                f"dim={dim}, keepdim=True{dtype_kwarg})"
            )
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, "
            f"dim={dim}{dtype_kwarg})"
        )


class SumOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.sum")

    @property
    def torch_op_name(self) -> str:
        return self.name


class MeanOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = FLOAT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.mean")

    @property
    def torch_op_name(self) -> str:
        return self.name


class ProdOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.prod")

    @property
    def torch_op_name(self) -> str:
        return self.name


class NansumOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = FLOAT_DTYPES
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.nansum")

    @property
    def torch_op_name(self) -> str:
        return self.name


class VarOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = FLOAT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.var")
        self._correction: int | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def _pick_scalar_input_shape(self) -> tuple[int, ...]:
        if self._correction == 0:
            return _random_input_shape()
        return _random_input_shape_min_numel_2()

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        self._correction = random.choice([0, 1, 2])
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        correction = self._correction
        self._correction = None
        base = super().codegen(output_name, input_names, output_spec)
        return base[:-1] + f", correction={correction})"


class AllOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = (torch.bool,)
    input_dtype_pool = _BOOL_OR_NUMERIC_INPUT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.all")

    @property
    def torch_op_name(self) -> str:
        return self.name


class AnyOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = (torch.bool,)
    input_dtype_pool = _BOOL_OR_NUMERIC_INPUT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.any")

    @property
    def torch_op_name(self) -> str:
        return self.name


class AmaxOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.amax")

    @property
    def torch_op_name(self) -> str:
        return self.name


class AminOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.amin")

    @property
    def torch_op_name(self) -> str:
        return self.name


class ArgmaxOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = (torch.int64,)
    input_dtype_pool = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.argmax")

    @property
    def torch_op_name(self) -> str:
        return self.name


class ArgminOperator(_FullOrKeepdimReductionBase):
    allowed_output_dtypes = (torch.int64,)
    input_dtype_pool = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.argmin")

    @property
    def torch_op_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Family D -- Dim-required tuple .values
# ---------------------------------------------------------------------------


class _DimRequiredTupleReductionBase(Operator):
    """2-arg-with-dim form, .values extraction, keepdim=True or False.

    Used for max, min, median, mode. When ``supports_keepdim_false``
    is False (default), ``can_produce`` requires a size-1 dim in the
    output (keepdim=True only). When True, any non-scalar shape is
    accepted and keepdim=False may be selected.
    """

    supports_keepdim_false: bool = False

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._dim: int | None = None
        self._keepdim: bool | None = None
        self._input_size: tuple[int, ...] | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        if not isinstance(output_spec, TensorSpec):
            return False
        if len(output_spec.size) < 1:
            return False
        if output_spec.dtype not in _NUMERIC_INPUT_DTYPES_NO_BOOL:
            return False
        if 1 in output_spec.size:
            return True
        return self.supports_keepdim_false

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        if 1 in output_spec.size and (
            not self.supports_keepdim_false or random.random() < 0.5
        ):
            self._keepdim = True
            size_one_dims = [i for i, s in enumerate(output_spec.size) if s == 1]
            self._dim = random.choice(size_one_dims)
            new_dim = random.randint(2, 6)
            self._input_size = tuple(
                new_dim if i == self._dim else s for i, s in enumerate(output_spec.size)
            )
        else:
            self._keepdim = False
            ndim = len(output_spec.size)
            self._dim = random.randint(0, ndim)
            r = random.randint(2, 6)
            input_size = list(output_spec.size)
            input_size.insert(self._dim, r)
            self._input_size = tuple(input_size)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        dim = self._dim
        keepdim = self._keepdim
        self._dim = None
        self._keepdim = None
        self._input_size = None
        if keepdim:
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}, "
                f"dim={dim}, keepdim=True).values"
            )
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim}).values"
        )


class MaxOperator(_DimRequiredTupleReductionBase):
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.max")

    @property
    def torch_op_name(self) -> str:
        return self.name


class MinOperator(_DimRequiredTupleReductionBase):
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.min")

    @property
    def torch_op_name(self) -> str:
        return self.name


class MedianOperator(_DimRequiredTupleReductionBase):
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.median")

    @property
    def torch_op_name(self) -> str:
        return self.name


class ModeOperator(_DimRequiredTupleReductionBase):
    supports_keepdim_false = True

    def __init__(self) -> None:
        super().__init__("torch.mode")

    @property
    def torch_op_name(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Family E -- Standalone special-case ops
# ---------------------------------------------------------------------------


class CountNonzeroOperator(Operator):
    """torch.count_nonzero -- 0-D int64 scalar output (no dim form)."""

    def __init__(self) -> None:
        super().__init__("torch.count_nonzero")
        self._input_size: tuple[int, ...] | None = None
        self._input_dtype: torch.dtype = _BOOL_OR_NUMERIC_INPUT_DTYPES[-1]

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and output_spec.dtype == torch.int64
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._input_size = _random_input_shape()
        self._input_dtype = random.choice(_BOOL_OR_NUMERIC_INPUT_DTYPES)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=self._input_dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        return f"{output_name} = torch.count_nonzero({input_names[0]})"


class AminmaxOperator(Operator):
    """torch.aminmax -- 0-D scalar via .min extraction (NamedTuple).

    Note: torch.aminmax DOES support a ``dim``/``keepdim`` form. This
    batch intentionally restricts coverage to the no-dim 0-D scalar form.
    """

    def __init__(self) -> None:
        super().__init__("torch.aminmax")
        self._input_size: tuple[int, ...] | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and output_spec.dtype in _NUMERIC_INPUT_DTYPES_NO_BOOL
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._input_size = _random_input_shape()
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        self._input_size = None
        return f"{output_name} = torch.aminmax({input_names[0]}).min"


class VarMeanOperator(Operator):
    """torch.var_mean -- 0-D float scalar via [0] extraction (NamedTuple).

    Note: torch.var_mean DOES support a ``dim``/``keepdim`` form. This
    batch intentionally restricts coverage to the no-dim 0-D scalar form.
    """

    def __init__(self) -> None:
        super().__init__("torch.var_mean")
        self._input_size: tuple[int, ...] | None = None
        self._correction: int | None = None

    @property
    def torch_op_name(self) -> str:
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and is_float_dtype(output_spec.dtype)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        assert isinstance(output_spec, TensorSpec)  # noqa: S101
        self._correction = random.choice([0, 1, 2])
        if self._correction == 0:
            self._input_size = _random_input_shape()
        else:
            self._input_size = _random_input_shape_min_numel_2()
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        correction = self._correction
        self._input_size = None
        self._correction = None
        # [0] extracts the variance from the (var, mean) tuple;
        # named-tuple .var access breaks under torch.compile tracing.
        return (
            f"{output_name} = torch.var_mean("
            f"{input_names[0]}, correction={correction})[0]"
        )
