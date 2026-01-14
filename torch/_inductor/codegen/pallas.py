from __future__ import annotations

import hashlib
import math
import typing_extensions
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet
from torch.utils._pallas import has_tpu_pallas
from torch.utils._sympy.functions import ModularIndexing

from .. import config
from ..ir import ComputedBuffer
from ..runtime.runtime_utils import torch_dtype_to_jax
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .block_analysis import BlockPatternMatcher
from .common import (
    BackendFeature,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import SIMDKernel, SIMDScheduling


class PallasPrinter(PythonPrinter):
    """
    Custom sympy printer for Pallas that handles JAX-specific constructs.
    """

    def _print_Where(self, expr: sympy.Expr) -> str:
        """Convert sympy Where to jnp.where."""
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"jnp.where({c}, {p}, {q})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        """Convert sympy Min to jnp.minimum for JAX compatibility."""
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.minimum({result}, {arg})"
        return result

    def _print_Max(self, expr: sympy.Expr) -> str:
        """Convert sympy Max to jnp.maximum for JAX compatibility."""
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.maximum({result}, {arg})"
        return result


# Use Pallas-specific printer for expression generation
pallas_pexpr = PallasPrinter().doprint


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ir import IRNode
    from ..ops_handler import ReductionType
    from ..scheduler import BaseSchedulerNode


# Main function suffix used in generated Pallas code
MAIN_SUFFIX = "main"

# Mosaic GPU warpgroup size: 4 warps × 32 threads = 128 threads per warpgroup.
# This is a hardware constant for Hopper and Blackwell GPUs.
# See: jax/_src/pallas/mosaic_gpu/lowering.py
WARPGROUP_SIZE = 128


def _align_to_warpgroup(size: int) -> int:
    """Align size to WARPGROUP_SIZE (128) for Mosaic GPU compatibility."""
    return ((size + WARPGROUP_SIZE - 1) // WARPGROUP_SIZE) * WARPGROUP_SIZE


# Logger for Pallas kernel code
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class PallasKernelWrapper:
    """Wrapper to provide .run() interface for Pallas kernels"""

    def __init__(
        self, kernel_fn: Callable[..., Any], kernel_path: Optional[str] = None
    ):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("Pallas kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        """
        Execute the Pallas kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


class Unsupported(RuntimeError):
    """Exception raised when an operation is not supported by the Pallas backend."""


class PallasKernelOverrides(OpOverrides):
    """
    Map element-wise ops to JAX/Pallas operations.

    For now, we use the default Python operators which are compatible
    with JAX numpy broadcasting semantics.
    """

    # Binary operations with on-demand reshape for broadcast compatibility
    @staticmethod
    def add(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} + {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def sub(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} - {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def mul(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} * {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def truediv(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} / {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def floordiv(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} // {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def mod(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"{a} % {b}"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def sin(x: str) -> str:
        return f"jnp.sin({x})"

    @staticmethod
    def cos(x: str) -> str:
        return f"jnp.cos({x})"

    @staticmethod
    def tan(x: str) -> str:
        return f"jnp.tan({x})"

    @staticmethod
    def sinh(x: str) -> str:
        return f"jnp.sinh({x})"

    @staticmethod
    def cosh(x: str) -> str:
        return f"jnp.cosh({x})"

    @staticmethod
    def tanh(x: str) -> str:
        return f"jnp.tanh({x})"

    @staticmethod
    def asin(x: str) -> str:
        return f"jnp.arcsin({x})"

    @staticmethod
    def acos(x: str) -> str:
        return f"jnp.arccos({x})"

    @staticmethod
    def atan(x: str) -> str:
        return f"jnp.arctan({x})"

    @staticmethod
    def exp(x: str) -> str:
        result = f"jnp.exp({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def exp2(x: str) -> str:
        result = f"jnp.exp2({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def expm1(x: str) -> str:
        result = f"jnp.expm1({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def log(x: str) -> str:
        result = f"jnp.log({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def log10(x: str) -> str:
        result = f"jnp.log10({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def log2(x: str) -> str:
        result = f"jnp.log2({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def log1p(x: str) -> str:
        result = f"jnp.log1p({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def sqrt(x: str) -> str:
        result = f"jnp.sqrt({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def rsqrt(x: str) -> str:
        result = f"(1.0 / jnp.sqrt({x}))"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def abs(x: str) -> str:
        result = f"jnp.abs({x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def neg(x: str) -> str:
        result = f"(-{x})"
        V.kernel._track_unary_op_shape(result, x)
        return result

    @staticmethod
    def floor(x: str) -> str:
        return f"jnp.floor({x})"

    @staticmethod
    def ceil(x: str) -> str:
        return f"jnp.ceil({x})"

    @staticmethod
    def trunc(x: str) -> str:
        return f"jnp.trunc({x})"

    @staticmethod
    def round(x: str) -> str:
        return f"jnp.round({x})"

    @staticmethod
    def sigmoid(x: str) -> str:
        return f"(1.0 / (1.0 + jnp.exp(-{x})))"

    @staticmethod
    def relu(x: str) -> str:
        return f"jnp.maximum({x}, 0)"

    @staticmethod
    def pow(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"jnp.power({a}, {b})"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def maximum(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"jnp.maximum({a}, {b})"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def minimum(a: str, b: str) -> str:
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        result = f"jnp.minimum({a}, {b})"
        V.kernel._track_binary_op_shape(result, a, b)
        return result

    @staticmethod
    def where(cond: str, a: str, b: str) -> str:
        # Ensure all three operands are broadcast compatible
        # First make a and b compatible
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        # Then make cond compatible with the result of a/b
        cond, a = V.kernel._ensure_broadcast_compatible(cond, a)
        # After making cond compatible with a, check a and b again
        a, b = V.kernel._ensure_broadcast_compatible(a, b)
        # Also check cond with b to be sure
        cond, b = V.kernel._ensure_broadcast_compatible(cond, b)
        return f"jnp.where({cond}, {a}, {b})"

    @staticmethod
    def masked(mask: str, body: Callable[[], str], other: float) -> str:
        """
        Computes body, but only uses the result where mask is true.
        Where mask is false, uses the 'other' value instead.
        """
        result = body()
        # Format the 'other' value properly for JAX
        if isinstance(other, float):
            if math.isnan(other):
                other_str = "jnp.nan"
            elif math.isinf(other):
                other_str = "jnp.inf" if other > 0 else "-jnp.inf"
            else:
                other_str = repr(other)
        else:
            other_str = repr(other)
        # Use jnp.where to select between result and other based on mask
        return f"jnp.where({mask}, {result}, {other_str})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> str:
        jax_dtype = torch_dtype_to_jax(dtype)
        # Wrap in jnp.asarray to handle scalars from integer indexing
        return f"jnp.asarray({x}).astype({jax_dtype})"

    @staticmethod
    def to_dtype_bitcast(x: str, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        """Bitcast a value from one dtype to another with the same size."""
        jax_dtype = torch_dtype_to_jax(dtype)
        jax_src_dtype = torch_dtype_to_jax(src_dtype)
        # First ensure the value is the correct source dtype, then bitcast
        return f"jax.lax.bitcast_convert_type(jnp.asarray({x}).astype({jax_src_dtype}), {jax_dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        """Convert a sympy expression to a JAX array indexing expression."""
        from ..utils import get_bounds_index_expr

        # Prepare and rename indexing to register size symbols as kernel args
        prepared = V.kernel.prepare_indexing(expr)
        renamed = V.kernel.rename_indexing(prepared)
        idx_str = V.kernel.kexpr(renamed)
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return PallasKernelOverrides.to_dtype(var, dtype)

    @staticmethod
    def constant(val, dtype: torch.dtype) -> str:
        """Convert a constant value to JAX representation."""
        jax_dtype = torch_dtype_to_jax(dtype)
        if dtype == torch.bool:
            result = "True" if val else "False"
        elif isinstance(val, float):
            # Handle special float values
            if math.isnan(val):
                result = "jnp.nan"
            elif math.isinf(val):
                result = "jnp.inf" if val > 0 else "-jnp.inf"
            else:
                result = f"jnp.array({val}, dtype={jax_dtype})"
        else:
            result = f"jnp.array({val}, dtype={jax_dtype})"
        # Track scalar shape for constants
        V.kernel.var_shapes[result] = ()
        return result

    @staticmethod
    def real(x: str) -> str:
        return f"jnp.real({x})"

    @staticmethod
    def imag(x: str) -> str:
        return f"jnp.imag({x})"

    @staticmethod
    def conj(x: str) -> str:
        return f"jnp.conj({x})"

    @staticmethod
    def angle(x: str) -> str:
        return f"jnp.angle({x})"

    @staticmethod
    def view_as_real(x: str) -> str:
        """View complex tensor as real tensor with extra dimension."""
        return f"jnp.stack([jnp.real({x}), jnp.imag({x})], axis=-1)"

    @staticmethod
    def view_as_complex(x: str) -> str:
        """View real tensor as complex tensor."""
        return f"({x}[..., 0] + 1j * {x}[..., 1])"

    # Comparison operations
    @staticmethod
    def eq(a: str, b: str) -> str:
        return f"({a} == {b})"

    @staticmethod
    def ne(a: str, b: str) -> str:
        return f"({a} != {b})"

    @staticmethod
    def lt(a: str, b: str) -> str:
        return f"({a} < {b})"

    @staticmethod
    def le(a: str, b: str) -> str:
        return f"({a} <= {b})"

    @staticmethod
    def gt(a: str, b: str) -> str:
        return f"({a} > {b})"

    @staticmethod
    def isnan(x: str) -> str:
        return f"jnp.isnan({x})"

    @staticmethod
    def isinf(x: str) -> str:
        return f"jnp.isinf({x})"

    @staticmethod
    def isfinite(x: str) -> str:
        return f"jnp.isfinite({x})"

    @staticmethod
    def ge(a: str, b: str) -> str:
        return f"({a} >= {b})"

    # Logical operations
    @staticmethod
    def logical_and(a: str, b: str) -> str:
        return f"jnp.logical_and({a}, {b})"

    @staticmethod
    def logical_or(a: str, b: str) -> str:
        return f"jnp.logical_or({a}, {b})"

    @staticmethod
    def logical_not(x: str) -> str:
        return f"jnp.logical_not({x})"

    @staticmethod
    def logical_xor(a: str, b: str) -> str:
        return f"jnp.logical_xor({a}, {b})"

    # Math operations
    @staticmethod
    def atan2(a: str, b: str) -> str:
        return f"jnp.arctan2({a}, {b})"

    @staticmethod
    def hypot(a: str, b: str) -> str:
        return f"jnp.hypot({a}, {b})"

    @staticmethod
    def fmod(a: str, b: str) -> str:
        return f"jnp.fmod({a}, {b})"

    @staticmethod
    def remainder(a: str, b: str) -> str:
        return f"jnp.remainder({a}, {b})"

    @staticmethod
    def truncdiv(a: str, b: str) -> str:
        # Truncated division (rounds toward zero)
        # For integers: sign(a)*sign(b) * (abs(a) // abs(b))
        return f"(jnp.sign({a}) * jnp.sign({b}) * (jnp.abs({a}) // jnp.abs({b}))).astype({a}.dtype)"

    @staticmethod
    def floordiv(a: str, b: str) -> str:
        return f"({a} // {b})"

    @staticmethod
    def clamp(x: str, min_val: str, max_val: str) -> str:
        return f"jnp.clip({x}, {min_val}, {max_val})"

    @staticmethod
    def clip(x: str, min_val: str, max_val: str) -> str:
        return f"jnp.clip({x}, {min_val}, {max_val})"

    # Sign operations
    @staticmethod
    def sign(x: str) -> str:
        return f"jnp.sign({x})"

    @staticmethod
    def signbit(x: str) -> str:
        return f"jnp.signbit({x})"

    # Special math functions
    @staticmethod
    def erf(x: str) -> str:
        return f"jax.scipy.special.erf({x})"

    @staticmethod
    def erfc(x: str) -> str:
        return f"jax.scipy.special.erfc({x})"

    @staticmethod
    def erfinv(x: str) -> str:
        return f"jax.scipy.special.erfinv({x})"

    @staticmethod
    def lgamma(x: str) -> str:
        return f"jax.scipy.special.gammaln({x})"

    @staticmethod
    def digamma(x: str) -> str:
        return f"jax.scipy.special.digamma({x})"

    @staticmethod
    def bessel_j0(x: str) -> str:
        # bessel_jn requires float64 and has numerical issues at x=0 (returns NaN)
        # bessel_jn(x, v=n) returns array of shape (n+1, ...) with J_0 to J_n
        # Handle by: convert to float64, compute, handle x=0, convert back
        # J0(0) = 1.0
        return (
            f"jnp.where({x}.astype(jnp.float64) == 0.0, 1.0, "
            f"jax.scipy.special.bessel_jn({x}.astype(jnp.float64), v=0)[0])"
            f".astype({x}.dtype)"
        )

    @staticmethod
    def bessel_j1(x: str) -> str:
        # bessel_jn requires float64 and has numerical issues at x=0 (returns NaN)
        # bessel_jn(x, v=n) returns array of shape (n+1, ...) with J_0 to J_n
        # Handle by: convert to float64, compute, handle x=0, convert back
        # J1(0) = 0.0
        return (
            f"jnp.where({x}.astype(jnp.float64) == 0.0, 0.0, "
            f"jax.scipy.special.bessel_jn({x}.astype(jnp.float64), v=1)[1])"
            f".astype({x}.dtype)"
        )

    @staticmethod
    def modified_bessel_i0(x: str) -> str:
        # Modified Bessel function of the first kind I_0(x)
        # I_0(x) = bessel_i0e(x) * exp(|x|) where bessel_i0e is the scaled version
        return f"jax.lax.bessel_i0e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def modified_bessel_i1(x: str) -> str:
        # Modified Bessel function of the first kind I_1(x)
        # I_1(x) = bessel_i1e(x) * exp(|x|) where bessel_i1e is the scaled version
        return f"jax.lax.bessel_i1e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def spherical_bessel_j0(x: str) -> str:
        # Spherical Bessel function of the first kind j_0(x) = sin(x) / x
        # Handle x=0: j_0(0) = 1
        return f"jnp.where({x} == 0.0, 1.0, jnp.sin({x}) / {x})"

    @staticmethod
    def i0(x: str) -> str:
        # Modified Bessel function I_0 (same as modified_bessel_i0)
        return f"jax.lax.bessel_i0e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def i0e(x: str) -> str:
        # Exponentially scaled modified Bessel function I_0
        return f"jax.lax.bessel_i0e({x})"

    @staticmethod
    def i1(x: str) -> str:
        # Modified Bessel function I_1 (same as modified_bessel_i1)
        return f"jax.lax.bessel_i1e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def i1e(x: str) -> str:
        # Exponentially scaled modified Bessel function I_1
        return f"jax.lax.bessel_i1e({x})"

    @staticmethod
    def gammainc(x: str, y: str) -> str:
        # Regularized lower incomplete gamma function P(a, x)
        # Note: PyTorch uses gammainc(input, other) where input is a (shape param)
        return f"jax.scipy.special.gammainc({x}, {y})"

    @staticmethod
    def gammaincc(x: str, y: str) -> str:
        # Regularized upper incomplete gamma function Q(a, x)
        return f"jax.scipy.special.gammaincc({x}, {y})"

    @staticmethod
    def igamma(x: str, y: str) -> str:
        # Regularized lower incomplete gamma function (alias for gammainc)
        return f"jax.scipy.special.gammainc({x}, {y})"

    @staticmethod
    def igammac(x: str, y: str) -> str:
        # Regularized upper incomplete gamma function (alias for gammaincc)
        return f"jax.scipy.special.gammaincc({x}, {y})"

    @staticmethod
    def polygamma(x: str, y: str) -> str:
        # Polygamma function psi^(n)(x), x is order n, y is the value
        # Note: JAX uses polygamma(n, x) where n is integer order
        return f"jax.scipy.special.polygamma({x}.astype(jnp.int32), {y})"

    @staticmethod
    def ndtri(x: str) -> str:
        # Inverse of the standard normal CDF
        return f"jax.scipy.special.ndtri({x})"

    @staticmethod
    def zeta(x: str, y: str) -> str:
        # Hurwitz zeta function zeta(x, q) = sum_{k=0}^inf 1/(k+q)^x
        return f"jax.scipy.special.zeta({x}, {y})"

    @staticmethod
    def xlogy(x: str, y: str) -> str:
        # x * log(y), with proper handling of x=0
        return f"jax.scipy.special.xlogy({x}, {y})"

    @staticmethod
    def xlog1py(x: str, y: str) -> str:
        # x * log1p(y), with proper handling of x=0
        return f"jax.scipy.special.xlog1py({x}, {y})"

    @staticmethod
    def chebyshev_polynomial_t(x: str, n: str) -> str:
        # Chebyshev polynomial of the first kind T_n(x)
        # For |x| <= 1: T_n(x) = cos(n * arccos(x))
        # For x > 1: T_n(x) = cosh(n * arccosh(x))
        # For x < -1: T_n(x) = (-1)^n * cosh(n * arccosh(-x))
        return (
            f"jnp.where(jnp.abs({x}) <= 1, "
            f"jnp.cos({n} * jnp.arccos(jnp.clip({x}, -1, 1))), "
            f"jnp.where({x} > 1, "
            f"jnp.cosh({n} * jnp.arccosh(jnp.maximum({x}, 1.0))), "
            f"((-1.0) ** {n}) * jnp.cosh({n} * jnp.arccosh(jnp.maximum(-{x}, 1.0)))))"
        )

    @staticmethod
    def chebyshev_polynomial_u(x: str, n: str) -> str:
        # Chebyshev polynomial of the second kind U_n(x)
        # For |x| < 1: U_n(x) = sin((n+1) * arccos(x)) / sqrt(1 - x^2)
        # For x = 1: U_n(1) = n+1
        # For x = -1: U_n(-1) = (-1)^n * (n+1)
        # For x > 1: U_n(x) = sinh((n+1) * arccosh(x)) / sqrt(x^2 - 1)
        # For x < -1: U_n(x) = (-1)^n * U_n(-x) (symmetry)
        return (
            f"jnp.where(jnp.abs({x}) < 1, "
            f"jnp.sin(({n} + 1) * jnp.arccos(jnp.clip({x}, -1, 1))) / "
            f"jnp.sqrt(jnp.maximum(1 - {x}**2, 1e-10)), "
            f"jnp.where({x} >= 1, "
            f"jnp.where({x} == 1, {n} + 1.0, "
            f"jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum({x}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({x}**2 - 1, 1e-10))), "
            f"jnp.where({x} == -1, ((-1.0) ** {n}) * ({n} + 1.0), "
            f"((-1.0) ** {n}) * jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum(-{x}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({x}**2 - 1, 1e-10)))))"
        )

    @staticmethod
    def chebyshev_polynomial_v(x: str, n: str) -> str:
        # Chebyshev polynomial of the third kind V_n(x)
        # V_n(x) = (T_n(x) - T_{n+1}(x)) / (1 - x) for x != 1
        # V_n(1) = 1, recurrence: V_0 = 1, V_1 = 2x - 1, V_n = 2x*V_{n-1} - V_{n-2}
        # Explicit: V_0 = 1, V_1 = 2x-1, V_2 = 4x^2-2x-1, V_3 = 8x^3-4x^2-4x+1
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{x} - 1, "
            f"jnp.where({n} == 2, 4*{x}**2 - 2*{x} - 1, "
            f"jnp.where({n} == 3, 8*{x}**3 - 4*{x}**2 - 4*{x} + 1, "
            f"jnp.where({n} == 4, 16*{x}**4 - 8*{x}**3 - 12*{x}**2 + 4*{x} + 1, "
            f"jnp.where({n} == 5, 32*{x}**5 - 16*{x}**4 - 32*{x}**3 + 12*{x}**2 + 6*{x} - 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def chebyshev_polynomial_w(x: str, n: str) -> str:
        # Chebyshev polynomial of the fourth kind W_n(x)
        # W_n(x) = (T_n(x) + T_{n+1}(x)) / (1 + x) for x != -1
        # W_n(-1) = (-1)^n, recurrence: W_0 = 1, W_1 = 2x + 1, W_n = 2x*W_{n-1} - W_{n-2}
        # Explicit: W_0 = 1, W_1 = 2x+1, W_2 = 4x^2+2x-1, W_3 = 8x^3+4x^2-4x-1
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{x} + 1, "
            f"jnp.where({n} == 2, 4*{x}**2 + 2*{x} - 1, "
            f"jnp.where({n} == 3, 8*{x}**3 + 4*{x}**2 - 4*{x} - 1, "
            f"jnp.where({n} == 4, 16*{x}**4 + 8*{x}**3 - 12*{x}**2 - 4*{x} + 1, "
            f"jnp.where({n} == 5, 32*{x}**5 + 16*{x}**4 - 32*{x}**3 - 12*{x}**2 + 6*{x} + 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_t(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the first kind T*_n(x) = T_n(2x - 1)
        # T_n(y) where y = 2x - 1
        # Use same formula as chebyshev_polynomial_t
        y = f"(2 * {x} - 1)"
        return (
            f"jnp.where(jnp.abs({y}) <= 1, "
            f"jnp.cos({n} * jnp.arccos(jnp.clip({y}, -1, 1))), "
            f"jnp.where({y} > 1, "
            f"jnp.cosh({n} * jnp.arccosh(jnp.maximum({y}, 1.0))), "
            f"((-1.0) ** {n}) * jnp.cosh({n} * jnp.arccosh(jnp.maximum(-{y}, 1.0)))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_u(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the second kind U*_n(x) = U_n(2x - 1)
        # Use same formula as chebyshev_polynomial_u
        y = f"(2 * {x} - 1)"
        return (
            f"jnp.where(jnp.abs({y}) < 1, "
            f"jnp.sin(({n} + 1) * jnp.arccos(jnp.clip({y}, -1, 1))) / "
            f"jnp.sqrt(jnp.maximum(1 - ({y})**2, 1e-10)), "
            f"jnp.where({y} >= 1, "
            f"jnp.where({y} == 1, {n} + 1.0, "
            f"jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum({y}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({y}**2 - 1, 1e-10))), "
            f"jnp.where({y} == -1, ((-1.0) ** {n}) * ({n} + 1.0), "
            f"((-1.0) ** {n}) * jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum(-{y}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({y}**2 - 1, 1e-10)))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_v(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the third kind V*_n(x) = V_n(2x - 1)
        y = f"(2 * {x} - 1)"  # shifted variable
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{y} - 1, "
            f"jnp.where({n} == 2, 4*{y}**2 - 2*{y} - 1, "
            f"jnp.where({n} == 3, 8*{y}**3 - 4*{y}**2 - 4*{y} + 1, "
            f"jnp.where({n} == 4, 16*{y}**4 - 8*{y}**3 - 12*{y}**2 + 4*{y} + 1, "
            f"jnp.where({n} == 5, 32*{y}**5 - 16*{y}**4 - 32*{y}**3 + 12*{y}**2 + 6*{y} - 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_w(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the fourth kind W*_n(x) = W_n(2x - 1)
        y = f"(2 * {x} - 1)"  # shifted variable
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{y} + 1, "
            f"jnp.where({n} == 2, 4*{y}**2 + 2*{y} - 1, "
            f"jnp.where({n} == 3, 8*{y}**3 + 4*{y}**2 - 4*{y} - 1, "
            f"jnp.where({n} == 4, 16*{y}**4 + 8*{y}**3 - 12*{y}**2 - 4*{y} + 1, "
            f"jnp.where({n} == 5, 32*{y}**5 + 16*{y}**4 - 32*{y}**3 - 12*{y}**2 + 6*{y} + 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def hermite_polynomial_h(x: str, n: str) -> str:
        # Physicist's Hermite polynomial H_n(x)
        # H_n(x) = 2^n * x^n - n*(n-1)/2 * 2^(n-2) * x^(n-2) + ...
        # Use explicit formula: H_n(x) = n! * sum_{m=0}^{n//2} (-1)^m / (m! * (n-2m)!) * (2x)^(n-2m)
        # For simplicity, use the relation: H_n(x) = 2^(n/2) * He_n(x * sqrt(2)) where He is probabilist's
        # Actually simpler: use recurrence or closed form
        # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2 * {x}, "
            f"jnp.where({n} == 2, 4 * {x}**2 - 2, "
            f"jnp.where({n} == 3, 8 * {x}**3 - 12 * {x}, "
            f"jnp.where({n} == 4, 16 * {x}**4 - 48 * {x}**2 + 12, "
            f"jnp.where({n} == 5, 32 * {x}**5 - 160 * {x}**3 + 120 * {x}, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def hermite_polynomial_he(x: str, n: str) -> str:
        # Probabilist's Hermite polynomial He_n(x)
        # He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, {x}, "
            f"jnp.where({n} == 2, {x}**2 - 1, "
            f"jnp.where({n} == 3, {x}**3 - 3 * {x}, "
            f"jnp.where({n} == 4, {x}**4 - 6 * {x}**2 + 3, "
            f"jnp.where({n} == 5, {x}**5 - 10 * {x}**3 + 15 * {x}, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def laguerre_polynomial_l(x: str, n: str) -> str:
        # Laguerre polynomial L_n(x)
        # L_0 = 1, L_1 = 1 - x, L_2 = (x^2 - 4x + 2)/2, L_3 = (-x^3 + 9x^2 - 18x + 6)/6
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 1 - {x}, "
            f"jnp.where({n} == 2, ({x}**2 - 4*{x} + 2) / 2, "
            f"jnp.where({n} == 3, (-{x}**3 + 9*{x}**2 - 18*{x} + 6) / 6, "
            f"jnp.where({n} == 4, ({x}**4 - 16*{x}**3 + 72*{x}**2 - 96*{x} + 24) / 24, "
            f"jnp.where({n} == 5, (-{x}**5 + 25*{x}**4 - 200*{x}**3 + 600*{x}**2 - 600*{x} + 120) / 120, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def legendre_polynomial_p(x: str, n: str) -> str:
        # Legendre polynomial P_n(x)
        # P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, P_3 = (5x^3 - 3x)/2
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, {x}, "
            f"jnp.where({n} == 2, (3 * {x}**2 - 1) / 2, "
            f"jnp.where({n} == 3, (5 * {x}**3 - 3 * {x}) / 2, "
            f"jnp.where({n} == 4, (35 * {x}**4 - 30 * {x}**2 + 3) / 8, "
            f"jnp.where({n} == 5, (63 * {x}**5 - 70 * {x}**3 + 15 * {x}) / 8, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    # Reciprocal and square
    @staticmethod
    def reciprocal(x: str) -> str:
        return f"jnp.reciprocal({x})"

    @staticmethod
    def square(x: str) -> str:
        return f"jnp.square({x})"

    # Additional operations
    @staticmethod
    def fma(a: str, b: str, c: str) -> str:
        """Fused multiply-add: a * b + c

        JAX doesn't have jnp.fma, so we use the unfused version.
        The compiler may still fuse this on supported hardware.
        """
        return f"(({a}) * ({b}) + ({c}))"

    @staticmethod
    def copysign(a: str, b: str) -> str:
        return f"jnp.copysign({a}, {b})"

    @staticmethod
    def nextafter(a: str, b: str) -> str:
        return f"jnp.nextafter({a}, {b})"

    @staticmethod
    def ldexp(a: str, b: str) -> str:
        return f"jnp.ldexp({a}, {b})"

    @staticmethod
    def frexp(x: str) -> str:
        return f"jnp.frexp({x})"

    @staticmethod
    def modf(x: str) -> str:
        return f"jnp.modf({x})"

    # Bitwise operations
    @staticmethod
    def bitwise_and(a: str, b: str) -> str:
        return f"jnp.bitwise_and({a}, {b})"

    @staticmethod
    def bitwise_or(a: str, b: str) -> str:
        return f"jnp.bitwise_or({a}, {b})"

    @staticmethod
    def bitwise_xor(a: str, b: str) -> str:
        return f"jnp.bitwise_xor({a}, {b})"

    @staticmethod
    def bitwise_not(x: str) -> str:
        return f"jnp.bitwise_not({x})"

    @staticmethod
    def left_shift(a: str, b: str) -> str:
        return f"jnp.left_shift({a}, {b})"

    @staticmethod
    def right_shift(a: str, b: str) -> str:
        return f"jnp.right_shift({a}, {b})"

    # Random number generation operations
    @staticmethod
    def load_seed(name: str, offset: str) -> str:
        """Load the random seed value from a buffer."""
        # Load the seed from the buffer and add offset for uniqueness
        seed_offset = V.kernel.args.seed_offset("load_seed_offset", offset)
        return f"({V.kernel.args.input(name)}[0] + {seed_offset})"

    @staticmethod
    def rand(seed: str, offset: str) -> str:
        """Generate uniform random numbers in [0, 1).

        Uses JAX's threefry2x32 PRNG directly for vectorized random generation.
        The seed provides the base key, offset provides per-element uniqueness.
        """
        # For vectorized random, we use jax.random.uniform with shape from offset
        # Create a base key from seed, then use fold_in with vmap for per-element keys
        # Use float32 dtype to match PyTorch's default
        return (
            f"jax.vmap(lambda o: jax.random.uniform("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), dtype=jnp.float32))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )

    @staticmethod
    def randn(seed: str, offset: str) -> str:
        """Generate standard normal random numbers.

        Uses JAX's threefry2x32 PRNG directly for vectorized random generation.
        The seed provides the base key, offset provides per-element uniqueness.
        """
        # For vectorized random, use vmap to fold in each offset value
        # Use float32 dtype to match PyTorch's default
        return (
            f"jax.vmap(lambda o: jax.random.normal("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), dtype=jnp.float32))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )

    @staticmethod
    def randint64(seed: str, offset: str, low: str, high: str) -> str:
        """Generate random int64 values in [low, high)."""
        # For vectorized random, use vmap to fold in each offset value
        return (
            f"jax.vmap(lambda o: jax.random.randint("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), {low}, {high}, dtype=jnp.int64))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )


class PallasKernel(SIMDKernel):
    """
    Pallas kernel for elementwise operations with support for strided/scatter access.

    Strategy:
    - Convert index expressions to JAX-compatible array slicing
    - Load/store using indexed access: "in_ptrX[slice]" or full-array "in_ptrX[...]"
    - Compute expression with Python operators (compatible with jax.numpy broadcasting)
    - Generate Python code that defines a Pallas kernel and a host entrypoint.
    - Use async_compile.pallas path to compile and load Python code.

    For GPU (Mosaic backend):
    - Use TMA (Tensor Memory Accelerator) for automatic OOB masking
    - Falls back to legacy padding approach for reductions, broadcasting, non-contiguous tensors
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pallas_pexpr  # Use Pallas expression printer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine device type once at initialization
        device = V.graph.get_current_device_or_throw()
        self.is_gpu = device.type == "cuda"
        # Use TMA (Tensor Memory Accelerator) for GPU to handle non-aligned tensor sizes
        # TMA automatically masks OOB accesses, eliminating the need for explicit
        # padding to multiples of 128. Uses lax.fori_loop with direct TMA primitives.
        self.use_emit_pipeline = self.is_gpu  # Enable TMA approach for GPU
        # Legacy: warpgroup padding (enabled when TMA approach is disabled)
        self.use_warpgroup_padding = self.is_gpu and not self.use_emit_pipeline
        # Track which output param each store uses: list of (out_ptr_name, store_line)
        self.store_with_output: list[tuple[str, str]] = []
        # Track load index expressions for argmax/argmin axis detection
        self.load_index_exprs: dict[str, sympy.Expr] = {}
        # Track outputs that need to be readable (for scatter operations)
        self.outputs_need_read: OrderedSet[str] = OrderedSet()
        # Track if any load in this kernel used transpose
        # Used to avoid double transpose (load + store)
        self.has_transposed_load = False
        # Canonical output shape for this kernel (computed lazily)
        # Used to reshape intermediate buffers for proper broadcasting
        self._canonical_output_shape: Optional[list[int]] = None
        # Track shapes of CSE variables for on-demand reshape at binary ops
        # Maps variable name (e.g., "tmp0") to shape tuple (e.g., (2, 16, 64))
        self.var_shapes: dict[str, tuple[int, ...]] = {}

    def _get_canonical_output_shape(self) -> Optional[list[int]]:
        """
        Get the canonical output shape for this kernel.

        This uses the same principled logic as iteration variable emission:
        1. Find a buffer (output or input) whose numel matches an iteration var length
        2. That buffer's shape defines the N-D structure (reshape target)
        3. Combine with other iteration dimensions to get full canonical shape

        This is used to reshape intermediate buffers (e.g., (32, 64) -> (2, 16, 64))
        for proper broadcasting with other tensors in the kernel.
        """
        if self._canonical_output_shape is not None:
            return self._canonical_output_shape

        if not hasattr(self, "range_tree_nodes") or not self.range_tree_nodes:
            return None

        # Collect iteration variable lengths (same logic as iteration var emission)
        iter_lengths = set()
        reduction_lengths = []
        pointwise_lengths = []
        for var, entry in self.range_tree_nodes.items():
            length = self._safe_int(entry.length)
            if length is not None:
                iter_lengths.add(length)
                if entry.is_reduction:
                    reduction_lengths.append(length)
                else:
                    pointwise_lengths.append(length)

        # Helper to check if buffer's numel matches an iteration var
        def _get_nd_shape_if_matches(buf_name, prefer_no_ones=True):
            # Use get_buffer which works for both intermediate buffers and graph inputs
            buf = V.graph.get_buffer(buf_name)
            if buf is None or len(buf.get_size()) <= 1:
                return None, None
            shape = [self._safe_int(s) for s in buf.get_size()]
            if None in shape:
                return None, None
            numel = 1
            for s in shape:
                numel *= s
            if numel not in iter_lengths:
                return None, None
            # If prefer_no_ones, skip shapes with 1 dimensions (keepdims artifacts)
            # This includes interior 1s (e.g., [2, 1, 64]) and trailing 1s (e.g., [2, 16, 1])
            if prefer_no_ones and 1 in shape:
                return None, None
            return (shape, numel)

        # Find reshape target: buffer whose numel matches an iteration var
        # Check outputs first, then inputs (same order as iteration var emission)
        # Prefer buffers without interior 1 dimensions (to avoid keepdims artifacts)
        reshape_target_shape = None
        reshape_target_numel = None

        # First pass: prefer shapes without interior 1 dimensions
        for out_name in self.args.output_buffers:
            shape, numel = _get_nd_shape_if_matches(out_name, prefer_no_ones=True)
            if shape:
                reshape_target_shape, reshape_target_numel = shape, numel
                break

        if reshape_target_shape is None:
            for in_name in self.args.input_buffers:
                shape, numel = _get_nd_shape_if_matches(in_name, prefer_no_ones=True)
                if shape:
                    reshape_target_shape, reshape_target_numel = shape, numel
                    break

        # Second pass: accept shapes with interior 1 dimensions as fallback
        if reshape_target_shape is None:
            for out_name in self.args.output_buffers:
                shape, numel = _get_nd_shape_if_matches(out_name, prefer_no_ones=False)
                if shape:
                    # Strip interior 1s for the canonical shape
                    reshape_target_shape = [s for s in shape if s != 1]
                    if len(reshape_target_shape) < len(shape):
                        # Recompute numel without 1s
                        reshape_target_numel = 1
                        for s in reshape_target_shape:
                            reshape_target_numel *= s
                    else:
                        reshape_target_numel = numel
                    break

        if reshape_target_shape is None:
            for in_name in self.args.input_buffers:
                shape, numel = _get_nd_shape_if_matches(in_name, prefer_no_ones=False)
                if shape:
                    # Strip interior 1s for the canonical shape
                    reshape_target_shape = [s for s in shape if s != 1]
                    if len(reshape_target_shape) < len(shape):
                        reshape_target_numel = 1
                        for s in reshape_target_shape:
                            reshape_target_numel *= s
                    else:
                        reshape_target_numel = numel
                    break

        if reshape_target_shape is None:
            return None

        # Build canonical shape: reshape_target + other iteration dims
        # The reshape target replaces one iteration var, add the rest
        # NOTE: We should NOT append reduction lengths of 1 (keepdims artifacts)
        output_shape = list(reshape_target_shape)
        for length in pointwise_lengths:
            if length != reshape_target_numel:
                output_shape.append(length)
        for length in reduction_lengths:
            if length != reshape_target_numel and length != 1:
                # Skip length=1 which is typically from keepdims reduction
                output_shape.append(length)

        self._canonical_output_shape = output_shape
        return self._canonical_output_shape

    def _ensure_broadcast_compatible(self, a: str, b: str) -> tuple[str, str]:
        """
        Ensure two operands have broadcast-compatible shapes for JAX.

        JAX has stricter broadcasting rules than NumPy/PyTorch. This method
        checks if the shapes are compatible and reshapes if needed.

        For example:
        - (32, 64) and (2, 16, 64): reshape (32, 64) -> (2, 16, 64)
        - (2, 16, 64) and (2, 16, 1, 64): reshape (2, 16, 64) -> (2, 16, 1, 64)

        Returns the (possibly wrapped) operand strings.
        """
        shape_a = self.var_shapes.get(str(a))
        shape_b = self.var_shapes.get(str(b))

        # Try to infer shape from expression if not tracked
        if shape_a is None:
            shape_a = self._infer_shape_from_expr(str(a))
        if shape_b is None:
            shape_b = self._infer_shape_from_expr(str(b))

        # If we still don't know both shapes, return as-is
        if shape_a is None or shape_b is None:
            return a, b

        # If shapes are identical, no action needed
        if shape_a == shape_b:
            return a, b

        # Scalars (shape ()) naturally broadcast - don't try to reshape them
        # jnp.inf/jnp.nan are Python floats and can't be reshaped
        if len(shape_a) == 0 or len(shape_b) == 0:
            return a, b

        # Check if shapes already broadcast (same rank and compatible dims)
        if len(shape_a) == len(shape_b):
            compatible = True
            for sa, sb in zip(shape_a, shape_b):
                if sa != sb and sa != 1 and sb != 1:
                    compatible = False
                    break
            if compatible:
                return a, b

        # Compute numels
        numel_a = 1
        for s in shape_a:
            numel_a *= s
        numel_b = 1
        for s in shape_b:
            numel_b *= s

        # If numels match, we can reshape one to match the other
        if numel_a == numel_b:
            # IMPORTANT: Check for transpose case FIRST!
            # If shapes are the same dims but reversed, we need transpose, NOT reshape.
            # reshape() just reinterprets flat data; transpose() actually reorders elements.
            # E.g., (N, M) -> (M, N) requires transpose, not reshape.
            if (
                len(shape_a) == len(shape_b) == 2
                and sorted(shape_a) == sorted(shape_b)
                and list(shape_a) == list(reversed(shape_b))
            ):
                # shape_a is transpose of shape_b - transpose a to match b
                new_a = f"jnp.transpose({a})"
                self.var_shapes[new_a] = shape_b
                self.has_transposed_load = True
                return new_a, b

            # First priority: prefer higher dimensionality for safe JAX broadcasting
            # This ensures intermediate results (e.g., from partial_reduce with keepdims)
            # maintain their full shape rather than being collapsed
            if len(shape_a) > len(shape_b):
                # Reshape b to match a's higher dimensionality
                shape_str = ", ".join(str(s) for s in shape_a)
                new_b = f"{b}.reshape({shape_str})"
                self.var_shapes[new_b] = shape_a
                return a, new_b
            elif len(shape_b) > len(shape_a):
                # Reshape a to match b's higher dimensionality
                shape_str = ", ".join(str(s) for s in shape_b)
                new_a = f"{a}.reshape({shape_str})"
                self.var_shapes[new_a] = shape_b
                return new_a, b

            # Same dimensionality - use intermediate 1s count as tiebreaker
            # to avoid keepdims artifacts like (2, 16, 1, 64) vs (2, 16, 64)
            def count_intermediate_ones(shape):
                if len(shape) <= 2:
                    return 0
                # Count 1s that are not at the start or end
                middle = shape[1:-1] if len(shape) > 2 else []
                return sum(1 for s in middle if s == 1)

            ones_a = count_intermediate_ones(shape_a)
            ones_b = count_intermediate_ones(shape_b)

            if ones_a < ones_b:
                # Reshape b to match a's shape (a has fewer intermediate 1s)
                shape_str = ", ".join(str(s) for s in shape_a)
                new_b = f"{b}.reshape({shape_str})"
                self.var_shapes[new_b] = shape_a
                return a, new_b
            else:
                # Reshape a to match b's shape (or they're equal, default to b)
                shape_str = ", ".join(str(s) for s in shape_b)
                new_a = f"{a}.reshape({shape_str})"
                self.var_shapes[new_a] = shape_b
                return new_a, b

        # If one is broadcastable to the other (different numels)
        # This handles cases like (2, 16, 1) broadcasting with (2, 16, 64)
        # IMPORTANT: Try LEADING 1s FIRST to match PyTorch/NumPy broadcasting semantics.
        # PyTorch broadcasting aligns dimensions from the right, so (16,) broadcasting
        # with (16, 16) should become (1, 16) NOT (16, 1).
        if len(shape_a) < len(shape_b):
            n_pad = len(shape_b) - len(shape_a)
            # Try leading 1s first (standard broadcasting - dimensions align from right)
            padded_a_leading = (1,) * n_pad + shape_a
            compatible = all(
                sa == sb or sa == 1 or sb == 1
                for sa, sb in zip(padded_a_leading, shape_b)
            )
            if compatible:
                shape_str = ", ".join(str(s) for s in padded_a_leading)
                new_a = f"{a}.reshape({shape_str})"
                self.var_shapes[new_a] = padded_a_leading
                return new_a, b

            # Try trailing 1s (e.g., (2,16,1) -> (2,16,1,1) for (2,16,1,64))
            padded_a_trailing = shape_a + (1,) * n_pad
            compatible = all(
                sa == sb or sa == 1 or sb == 1
                for sa, sb in zip(padded_a_trailing, shape_b)
            )
            if compatible:
                shape_str = ", ".join(str(s) for s in padded_a_trailing)
                new_a = f"{a}.reshape({shape_str})"
                self.var_shapes[new_a] = padded_a_trailing
                return new_a, b

        elif len(shape_b) < len(shape_a):
            n_pad = len(shape_a) - len(shape_b)
            # Try leading 1s first (standard broadcasting - dimensions align from right)
            # E.g., (16,) with (16, 16) -> (16,) becomes (1, 16) to broadcast along rows
            padded_b_leading = (1,) * n_pad + shape_b
            compatible = all(
                sa == sb or sa == 1 or sb == 1
                for sa, sb in zip(shape_a, padded_b_leading)
            )
            if compatible:
                shape_str = ", ".join(str(s) for s in padded_b_leading)
                new_b = f"{b}.reshape({shape_str})"
                self.var_shapes[new_b] = padded_b_leading
                return a, new_b

            # Try trailing 1s as fallback
            padded_b_trailing = shape_b + (1,) * n_pad
            compatible = all(
                sa == sb or sa == 1 or sb == 1
                for sa, sb in zip(shape_a, padded_b_trailing)
            )
            if compatible:
                shape_str = ", ".join(str(s) for s in padded_b_trailing)
                new_b = f"{b}.reshape({shape_str})"
                self.var_shapes[new_b] = padded_b_trailing
                return a, new_b

        # Handle keepdims-style broadcasting: one shape has trailing 1s and
        # the non-1 product matches a prefix product of the other shape.
        # Example: (2, 16, 64) and (32, 1) -> reshape (32, 1) to (2, 16, 1)
        # because 32 = 2 * 16 (prefix product)
        def try_keepdims_reshape(
            target_shape: tuple[int, ...], keepdims_shape: tuple[int, ...], operand: str
        ) -> Optional[tuple[str, tuple[int, ...]]]:
            """Try to reshape keepdims_shape to match target_shape's prefix."""
            # Count trailing 1s in keepdims_shape
            n_trailing_ones = 0
            for d in reversed(keepdims_shape):
                if d == 1:
                    n_trailing_ones += 1
                else:
                    break
            if n_trailing_ones == 0:
                return None  # No trailing 1s, not a keepdims shape

            # Get the non-1 product
            non_one_dims = keepdims_shape[: len(keepdims_shape) - n_trailing_ones]
            keepdims_numel = 1
            for d in non_one_dims:
                keepdims_numel *= d

            # Try to match prefix products of target_shape
            prefix_product = 1
            for i, d in enumerate(target_shape):
                prefix_product *= d
                if prefix_product == keepdims_numel:
                    # Found matching prefix!
                    # New shape is target_shape[:i+1] + (1,) * (len(target_shape) - i - 1)
                    new_shape = target_shape[: i + 1] + (1,) * (len(target_shape) - i - 1)
                    return operand, new_shape
            return None

        # Try to reshape shape_b to match shape_a (keepdims case)
        result = try_keepdims_reshape(shape_a, shape_b, b)
        if result:
            operand, new_shape = result
            shape_str = ", ".join(str(s) for s in new_shape)
            new_b = f"{operand}.reshape({shape_str})"
            self.var_shapes[new_b] = new_shape
            return a, new_b

        # Try to reshape shape_a to match shape_b (keepdims case)
        result = try_keepdims_reshape(shape_b, shape_a, a)
        if result:
            operand, new_shape = result
            shape_str = ", ".join(str(s) for s in new_shape)
            new_a = f"{operand}.reshape({shape_str})"
            self.var_shapes[new_a] = new_shape
            return new_a, b

        # Fall back to returning as-is
        return a, b

    def _infer_shape_from_expr(self, expr: str) -> Optional[tuple[int, ...]]:
        """
        Try to infer the shape of an expression from its string representation.

        Handles patterns like:
        - "in_ptr0[...]" -> buffer shape
        - "in_ptr0[...].reshape(2, 16, 64)" -> (2, 16, 64)
        - "tmp0.reshape(2, 16, 64)" -> (2, 16, 64)
        - "tmp3" -> look up in CSE cache to find original expression's shape
        - "jnp.array(256, dtype=...)" -> () (scalar)
        - "jnp.asarray(...).astype(...)" -> inherit from inner expression
        - Integer literals -> () (scalar)
        """
        import re

        # Check for reshape pattern
        match = re.search(r"\.reshape\(([^)]+)\)", expr)
        if match:
            try:
                dims_str = match.group(1)
                dims = tuple(int(d.strip()) for d in dims_str.split(","))
                return dims
            except (ValueError, AttributeError):
                pass

        # Check for scalar patterns - these all have shape ()
        # jnp.array(value, dtype=...) with a scalar value
        if re.match(r"^jnp\.array\([^,\[\]]+,\s*dtype=", expr):
            return ()
        # Pure integer literals
        if re.match(r"^-?\d+$", expr):
            return ()
        # Pure float literals
        if re.match(r"^-?\d+\.?\d*$", expr):
            return ()
        # JAX special values
        if expr in ("jnp.nan", "jnp.inf", "-jnp.inf", "True", "False"):
            return ()

        # Check for type cast patterns - inherit shape from inner expression
        # jnp.asarray(x).astype(dtype) -> shape of x
        match = re.match(r"^jnp\.asarray\(([^)]+)\)\.astype\(", expr)
        if match:
            inner_expr = match.group(1)
            inner_shape = self._infer_shape_from_expr(inner_expr)
            if inner_shape is not None:
                return inner_shape
            # Also check var_shapes directly
            inner_shape = self.var_shapes.get(inner_expr)
            if inner_shape is not None:
                return inner_shape

        # Check for simple CSE variable name like "tmp3"
        # Look up the original expression in CSE cache and get its shape
        if re.match(r"^tmp\d+$", expr):
            # Search CSE cache for the expression that generated this variable
            for cache_key, cse_var in self.cse._cache.items():
                if str(cse_var) == expr:
                    # Found the original expression, check if we have its shape
                    shape = self.var_shapes.get(cache_key)
                    if shape is not None:
                        # Also cache this for future lookups
                        self.var_shapes[expr] = shape
                        return shape
                    # Shape not directly tracked, try to infer from the expression
                    inferred = self._infer_shape_from_expr(cache_key)
                    if inferred is not None:
                        self.var_shapes[expr] = inferred
                        return inferred
                    break

        # Check for indexed access pattern like "in_ptr0[...].flatten()[idx]"
        # These have shape determined by the index expression broadcast, which is
        # complex to infer. Return None to avoid incorrect reshapes.
        if ".flatten()[" in expr:
            return None

        # Check for buffer access pattern like "in_ptr0[...]"
        match = re.match(r"(in_ptr\d+|buf\d+)\[", expr)
        if match:
            buf_name = match.group(1)
            # Look up in args to get actual buffer name
            for actual_name, arg_name in self.args.input_buffers.items():
                if arg_name == buf_name:
                    buf_obj = V.graph.try_get_buffer(actual_name)
                    if buf_obj is not None:
                        buf_size = buf_obj.get_size()
                        shape = tuple(self._safe_int(s) for s in buf_size)
                        if None not in shape:
                            return shape
                    break

        # Don't use canonical shape as fallback - it doesn't account for
        # intermediate shapes like keepdims reductions. Return None and
        # let the caller handle unknown shapes.
        return None

    def _track_binary_op_shape(
        self, result_expr: str, a: str, b: str
    ) -> None:
        """
        Track the shape of a binary operation result.

        The result shape is the broadcast of the two operand shapes.
        """
        shape_a = self.var_shapes.get(str(a))
        shape_b = self.var_shapes.get(str(b))

        if shape_a is None:
            shape_a = self._infer_shape_from_expr(str(a))
        if shape_b is None:
            shape_b = self._infer_shape_from_expr(str(b))

        if shape_a is None or shape_b is None:
            return

        # Compute broadcast shape
        result_shape = self._compute_broadcast_shape(shape_a, shape_b)
        if result_shape is not None:
            self.var_shapes[result_expr] = result_shape

    def _compute_broadcast_shape(
        self, shape_a: tuple[int, ...], shape_b: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        """
        Compute the broadcast shape of two shapes.

        Returns None if shapes are not broadcast-compatible.
        """
        # Pad shorter shape with leading 1s
        max_len = max(len(shape_a), len(shape_b))
        padded_a = (1,) * (max_len - len(shape_a)) + shape_a
        padded_b = (1,) * (max_len - len(shape_b)) + shape_b

        result = []
        for sa, sb in zip(padded_a, padded_b):
            if sa == sb:
                result.append(sa)
            elif sa == 1:
                result.append(sb)
            elif sb == 1:
                result.append(sa)
            else:
                # Incompatible - shapes don't broadcast
                return None

        return tuple(result)

    def _track_unary_op_shape(self, result_expr: str, x: str) -> None:
        """Track the shape of a unary operation result (same as input)."""
        shape = self.var_shapes.get(str(x))
        if shape is None:
            shape = self._infer_shape_from_expr(str(x))
        if shape is not None:
            self.var_shapes[result_expr] = shape

    def _compute_indexed_load_shape(
        self, index: sympy.Expr
    ) -> Optional[tuple[int, ...]]:
        """
        Compute the shape of an indexed load from the index expression broadcast.

        For indexed loads like `buf[...].flatten()[x0 + 2*x1]`, the result shape
        is determined by broadcasting the shapes of x0 and x1 (as reshaped arrays).

        The iteration variables are reshaped for broadcasting:
        - Variables with larger coefficients become earlier dimensions
        - Each variable contributes its length to one dimension
        - Singleton dimensions (1) are inserted for other positions

        Example: index = x0 + 2*x1 with x0.length=2, x1.length=6
        - x1 has larger coeff (2), so it's earlier: reshape to (6, 1)
        - x0 has smaller coeff (1), so it's later: reshape to (1, 2)
        - Broadcast result: (6, 2)

        For embedding lookups with indirect variables like `buf[...].flatten()[x0 + 64*tmp0]`:
        - tmp0 (indirect var) has shape from indices tensor, e.g., (2, 16)
        - x0 (iter var) has length 64
        - The result broadcasts to (2, 16, 64)
        """
        # Check for indirect variables - they contribute their shape to the result
        indirect_vars = self._get_indirect_vars(index)
        indirect_shape: list[int] = []
        has_indirect = bool(indirect_vars)
        if indirect_vars:
            # Get shape of first indirect variable (they should all have same shape)
            for indirect_var in indirect_vars:
                indirect_var_shape = self.var_shapes.get(str(indirect_var))
                if indirect_var_shape and len(indirect_var_shape) > 0:
                    indirect_shape = list(indirect_var_shape)
                    break

        used_vars = self._get_used_iter_vars(index)
        if not used_vars and not indirect_shape:
            # If we have indirect variables but couldn't find their shape,
            # fall back to canonical output shape (which captures all dimensions)
            if has_indirect:
                canonical = self._get_canonical_output_shape()
                if canonical is not None:
                    return tuple(canonical)
            return None

        # Get iteration variable lengths and sort by coefficient (descending)
        # Higher coefficient = earlier dimension in the output
        var_info = []
        for var in used_vars:
            if var in self.range_tree_nodes:
                entry = self.range_tree_nodes[var]
                if entry.is_reduction:
                    continue  # Skip reduction variables
                length = self._safe_int(entry.length)
                if length is None:
                    return None  # Dynamic shape, can't compute
                # Get coefficient of this var in the index
                coeff = index.coeff(var)
                if coeff == 0:
                    coeff = sympy.diff(index, var)
                try:
                    coeff_val = int(coeff) if coeff.is_number else 0
                except (TypeError, AttributeError):
                    coeff_val = 0
                var_info.append((var, length, coeff_val))

        # Sort by coefficient descending (larger coeff = earlier dimension)
        var_info.sort(key=lambda x: x[2], reverse=True)

        # The broadcast shape combines indirect var shape with iteration var lengths
        # Indirect var shape comes first (leading dimensions from indices tensor)
        # Iteration var lengths come after (e.g., embedding dimension)
        iter_shape = tuple(length for _, length, _ in var_info)

        if indirect_shape:
            # Combine: indirect shape (e.g., batch, seq) + iter shape (e.g., embed_dim)
            shape = tuple(indirect_shape) + iter_shape
        elif has_indirect:
            # Have indirect vars but couldn't find their shape
            # Fall back to canonical output shape (which captures all dimensions)
            canonical = self._get_canonical_output_shape()
            if canonical is not None:
                return tuple(canonical)
            # If no canonical shape, just use iteration variables
            if var_info:
                shape = iter_shape
            else:
                return None
        elif var_info:
            shape = iter_shape
        else:
            return None

        return shape

    def _track_var_shape(
        self, var_name: str, buf_name: str, load_expr: str, index: sympy.Expr
    ) -> None:
        """
        Track the shape of a loaded variable for on-demand reshape at binary ops.

        This determines the effective shape for the variable. For indirect indexing
        (embedding lookups), the effective shape is based on the indices shape, not
        the buffer shape.
        """
        buf_obj = V.graph.try_get_buffer(buf_name)
        if buf_obj is None:
            return

        # Check for trailing .reshape() first - this takes priority over index-based
        # shape inference since it's the final shape of the expression.
        # This handles cases like: buf[...].flatten()[idx].reshape(20, 1048576)
        if ".reshape(" in load_expr:
            import re

            match = re.search(r"\.reshape\(([^)]+)\)$", load_expr)
            if match:
                try:
                    dims_str = match.group(1)
                    dims = tuple(int(d.strip()) for d in dims_str.split(","))
                    self.var_shapes[var_name] = dims
                    return
                except (ValueError, AttributeError):
                    pass

        # For indexed loads without trailing reshape, compute shape from index broadcast
        # The shape is determined by the broadcast of all iteration variable shapes
        if ".flatten()[" in load_expr:
            shape = self._compute_indexed_load_shape(index)
            if shape is not None:
                self.var_shapes[var_name] = shape
            return

        # Get original buffer shape
        buf_size = buf_obj.get_size()
        buf_shape = tuple(self._safe_int(s) for s in buf_size)
        if None in buf_shape:
            return

        shape = buf_shape

        # Check if load_expr includes a reshape - extract the target shape
        if ".reshape(" in load_expr:
            # Extract reshape dimensions from the expression
            # e.g., "in_ptr0[...].reshape(2, 16, 64)" -> (2, 16, 64)
            import re

            match = re.search(r"\.reshape\(([^)]+)\)", load_expr)
            if match:
                try:
                    dims_str = match.group(1)
                    dims = tuple(int(d.strip()) for d in dims_str.split(","))
                    shape = dims
                except (ValueError, AttributeError):
                    pass
        elif "jnp.transpose(" in load_expr:
            # Check if load_expr includes a transpose - permute the shape
            # e.g., "jnp.transpose(in_ptr0[...], axes=(1, 0,))" -> permute (8, 16) to (16, 8)
            import re

            match = re.search(
                r"jnp\.transpose\([^,]+,\s*axes=\(([^)]+)\)", load_expr
            )
            if match:
                try:
                    axes_str = match.group(1)
                    axes = tuple(
                        int(a.strip().rstrip(","))
                        for a in axes_str.split(",")
                        if a.strip().rstrip(",")
                    )
                    # Permute the shape: axes[i] = which original dim goes to output dim i
                    shape = tuple(buf_shape[a] for a in axes)
                except (ValueError, IndexError):
                    pass
        else:
            # Check if this is an indirect load (embedding lookup)
            # For indirect indexing, the effective shape is the canonical output shape
            has_indirect = self._has_indirect_vars(index)
            if has_indirect:
                # For indirect indexing (e.g., embedding), use canonical output shape
                canonical = self._get_canonical_output_shape()
                if canonical is not None:
                    shape = tuple(canonical)
            else:
                # For direct indexing, check if buffer shape needs to match canonical
                # BUT preserve buffer shapes with singleton dimensions (e.g., (1, 10))
                # as these are meaningful for broadcasting.
                # ALSO preserve higher-dimensional buffer shapes (e.g., (65, 2, 2, 2) vs (65, 8))
                # as these are needed for correct broadcasting with keepdims-style reductions.
                canonical = self._get_canonical_output_shape()
                if canonical is not None and list(canonical) != list(buf_shape):
                    # Don't replace buffer shape if it has singleton dimensions
                    # These indicate explicit broadcast structure (e.g., (1, 10) for row broadcast)
                    has_singletons = 1 in buf_shape
                    # Don't replace buffer shape if it has more dimensions than canonical
                    # Higher-dimensional shapes are needed for keepdims-style broadcasting
                    # e.g., (65, 2, 2, 2) should not become (65, 8)
                    has_more_dims = len(buf_shape) > len(canonical)
                    if not has_singletons and not has_more_dims:
                        buf_numel = 1
                        for s in buf_shape:
                            buf_numel *= s
                        canonical_numel = 1
                        for s in canonical:
                            canonical_numel *= s
                        # If numels match, track canonical shape as the effective shape
                        if buf_numel == canonical_numel:
                            shape = tuple(canonical)

        self.var_shapes[var_name] = shape

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        """Check array bounds for indirect indexing."""
        # For now, skip explicit bounds checking as JAX/Pallas handles this internally
        # TODO: Implement explicit bounds checking with assertions if needed

    def _get_index_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expression to a string suitable for Pallas indexing.

        Pallas operates on full arrays, so we need to convert index expressions
        to JAX array slicing. For example:
        - x0 -> "..." (contiguous access, full array)
        - 2*x0 -> "::2" (strided access with stride 2)
        - 2*x0 + 1 -> "1::2" (strided access with offset 1, stride 2)

        Args:
            index: The indexing expression to convert

        Returns:
            The indexing string to use in generated code
        """
        # Prepare and simplify the index
        prepared_index = self.prepare_indexing(index)

        # Note: Block variable detection (im2col patterns) is handled in load()/store()
        # where we have access to buffer dimensions. We check the buffer size
        # against iteration variables there to detect gather patterns.

        # For simple single-symbol access (contiguous case), we can use [...]
        # which is more efficient as it operates on the entire array at once
        if isinstance(prepared_index, sympy.Symbol):
            return "..."
        elif prepared_index.is_Integer:
            # Scalar index
            return str(prepared_index)
        else:
            # Complex expression (strided/scatter access)
            # Try to extract stride and offset for common patterns
            return self._convert_to_jax_slice(prepared_index)

    def _convert_to_jax_slice(self, index: sympy.Expr) -> str:
        """
        Convert a sympy index expression to JAX slice notation.

        Handles common patterns like:
        - stride*var -> ::stride
        - stride*var + offset -> offset::stride

        For more complex patterns, falls back to explicit indexing.
        Uses BlockPatternMatcher for robust pattern matching.
        """
        # Get the iteration variables for this kernel
        if not self.range_trees:
            return "..."

        # Rename symbolic sizes to kernel parameter names upfront
        index = self.rename_indexing(index)

        # Check for ModularIndexing - this is NOT contiguous access
        # ModularIndexing is used for roll/wrap-around operations
        if index.has(ModularIndexing):
            # Generate actual index expression - iteration variables are already
            # defined as jnp.arange arrays, so we just convert to JAX code
            return self.kexpr(index)

        # Simplify the index
        index = V.graph.sizevars.simplify(index)
        # Find which iteration variable(s) are used
        used_vars = self._get_used_iter_vars(index)

        if len(used_vars) == 0:
            # No iteration variables, this is a constant index
            return str(index)
        elif len(used_vars) == 1:
            # Single iteration variable - try to extract stride and offset using BlockPatternMatcher
            var = next(iter(used_vars))

            # Get the subexpression involving this variable
            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)

            # Try to match affine pattern: stride * var
            stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)

            if stride is not None:
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)

                if stride < 0:
                    return self.kexpr(index)

                if offset == 0:
                    return "..."

                # Non-zero offset: check if we can use slice notation
                if stride != 1:
                    return self.kexpr(index)

                try:
                    offset_val = int(offset)
                    if offset_val < 0:
                        return self.kexpr(index)
                except (TypeError, ValueError):
                    return self.kexpr(index)

                return f"{self.kexpr(offset)}::1"
            else:
                # Couldn't match affine pattern, fall back to original logic
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)
                if offset == 0 and var_expr == var:
                    # Just the variable itself, unit stride
                    return "..."
        elif len(used_vars) > 1:
            # Multi-dimensional indexing
            # For contiguous multi-dim access, all terms should have unit stride
            all_unit_stride = True
            for var in used_vars:
                var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)
                stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
                if stride != 1:
                    all_unit_stride = False
                    break
            if all_unit_stride:
                # Contiguous multi-dimensional access
                return "..."
            else:
                # Strided multi-dimensional access
                # For most cases, inputs are made contiguous before passing to JAX,
                # so strided tensors become contiguous and we can use [...]
                # The buffer size check in load() handles im2col-like patterns
                return "..."

        # For complex cases, use [...] since inputs are made contiguous
        return "..."

    def _generate_strided_index(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for strided/complex indexing patterns.

        For expressions like `2 * x3 + 32 * x2 + 256 * x1 + 1024 * x0`, we generate
        code that computes the flattened index array using broadcasting.

        The iteration variables (x0, x1, x2, x3) are already defined as jnp.arange arrays
        in the kernel. We just need to convert the sympy expression to JAX code.
        """
        free_symbols = index.free_symbols
        iter_vars = self._get_iter_vars()

        # Check that all free symbols are iteration variables (no indirect vars)
        used_vars = free_symbols & iter_vars
        if used_vars != free_symbols:
            raise Unsupported(
                f"Pallas backend does not yet support mixed index pattern: {index}"
            )

        # Convert sympy expression to Python/JAX code string
        # The iteration variables are already defined as jnp.arange arrays
        index_str = self.kexpr(index)

        # Mark this as requiring flatten access
        return index_str

    def _generate_index_array(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for complex indexing patterns.
        Delegates to _generate_strided_index.
        """
        return self._generate_strided_index(index)

    def _get_iter_vars(self) -> OrderedSet:
        """Get the set of iteration variable symbols."""
        return OrderedSet(self.range_tree_nodes.keys())

    def _get_used_iter_vars(self, index: sympy.Expr) -> OrderedSet:
        """Get iteration variables used in an index expression."""
        return index.free_symbols & self._get_iter_vars()

    def _has_iteration_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains iteration variables."""
        return bool(self._get_used_iter_vars(index))

    def _get_indirect_vars(self, index: sympy.Expr) -> list[sympy.Symbol]:
        """Get list of indirect variable symbols (tmp*) in an index expression."""
        return [s for s in index.free_symbols if str(s).startswith("tmp")]

    def _has_indirect_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains indirect variables."""
        return len(self._get_indirect_vars(index)) > 0

    def _get_expected_output_shape(self) -> list:
        """Get the expected output shape from iteration variables.

        Iteration variables are shaped for broadcasting. For 2D outputs:
        - First var (e.g., y0) gets shape (1, N) - innermost dimension
        - Second var (e.g., x1) gets shape (M, 1) - outermost dimension
        The broadcast result is (M, N).
        """
        # Collect variable lengths
        var_items = list(self.range_tree_nodes.items())
        broadcast_vars = []
        for var_sym, entry in var_items:
            length = self._safe_int(entry.length)
            if length is not None:
                broadcast_vars.append(length)

        if len(broadcast_vars) <= 1:
            return broadcast_vars

        # For 2D case: variables are reshaped in reverse order
        # First var is innermost (last dim), second var is outermost (first dim)
        # So output shape is [second_var_length, first_var_length, ...]
        return list(reversed(broadcast_vars))

    def _get_transpose_axes(
        self, name: str, index: sympy.Expr
    ) -> Optional[tuple[int, ...]]:
        """
        Compute transpose axes needed for N-D buffer access.

        Returns the axes permutation if transpose is needed, None otherwise.
        For a transpose from input layout to output layout, we need to determine
        which input dimension corresponds to which output dimension based on
        the coefficient pattern in the index expression.

        The key insight: we match each iteration variable to an input dimension by:
        1. Matching the coefficient in the index expr to the buffer stride
        2. Verifying the variable's length equals the buffer size at that dimension

        If both conditions match for all variables and result in a non-identity
        permutation, we have a transpose.
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return None

        buf_size = buf_obj.get_size()
        ndim = len(buf_size)

        # Need at least 2D for transpose
        if ndim < 2:
            return None

        layout = getattr(buf_obj, "get_layout", lambda: None)()
        if layout is None:
            return None

        buf_stride = getattr(layout, "stride", None)
        if buf_stride is None or len(buf_stride) != ndim:
            return None

        # Get iteration variable info
        var_items = list(self.range_tree_nodes.items())
        if len(var_items) != ndim:
            return None

        # Skip for reduction variables
        if any(entry.is_reduction for _, entry in var_items):
            return None

        # Get buffer strides and sizes as integers
        buf_strides = [self._safe_int(s) for s in buf_stride]
        buf_sizes = [self._safe_int(s) for s in buf_size]
        if None in buf_strides or None in buf_sizes:
            return None

        # Skip if any dimension has size <= 1 (no meaningful transpose)
        if any(s <= 1 for s in buf_sizes):
            return None

        # Check if buffer strides form a valid transpose pattern.
        # A true transpose has strides that are a permutation of contiguous strides.
        # Non-contiguous slices (e.g., aliased buffers) have strides that don't fit this pattern.
        # For shape (A, B), contiguous strides are (B, 1).
        # A transposed (A, B) from (B, A).T has strides (1, A).
        # Non-contiguous slice might have strides like (C, 1) where C > B.
        expected_contiguous_strides = []
        stride = 1
        for i in range(ndim - 1, -1, -1):
            expected_contiguous_strides.insert(0, stride)
            stride *= buf_sizes[i]

        # Check if strides are a permutation of contiguous strides
        if sorted(buf_strides) != sorted(expected_contiguous_strides):
            # Strides don't form a valid transpose pattern - likely a non-contiguous slice
            return None

        # IMPORTANT: Use reversed var_items for coefficient extraction.
        # The LoopBody's _simplify_loops applies a variable reordering that reverses
        # the mapping between original and new iteration variables. To correctly match
        # index coefficients with buffer dimensions, we need to extract coefficients
        # in reversed order to compensate for this reordering.
        reversed_var_items = list(reversed(var_items))

        # Get iteration variable lengths (in reversed order to match coefficient extraction)
        var_lengths = [self._safe_int(entry.length) for _, entry in reversed_var_items]
        if None in var_lengths:
            return None

        index = V.graph.sizevars.simplify(index)

        def get_coefficient(expr, var):
            """Extract coefficient of var from expression."""
            if expr == var:
                return 1
            if expr.is_Add:
                for term in expr.args:
                    coeff = get_coefficient(term, var)
                    if coeff is not None:
                        return coeff
            if expr.is_Mul:
                coeff = 1
                has_var = False
                for factor in expr.args:
                    if factor == var:
                        has_var = True
                    elif factor.is_number:
                        coeff *= int(factor)
                if has_var:
                    return coeff
            return None

        # Extract coefficients for each iteration variable (in reversed order)
        coefficients = []
        for var, _entry in reversed_var_items:
            coeff = get_coefficient(index, var)
            if coeff is None:
                return None
            coefficients.append(coeff)

        # Build output_to_input mapping: for each output dim, which input dim does it read from
        # We require BOTH coefficient match (stride) AND length match (size) for validity
        output_to_input = []
        used_input_dims = set()

        for out_dim, (coeff, var_len) in enumerate(zip(coefficients, var_lengths)):
            # Find the input dimension whose stride matches this coefficient
            # AND whose size matches the variable length
            best_input_dim = None
            for in_dim, (stride, size) in enumerate(zip(buf_strides, buf_sizes)):
                if in_dim not in used_input_dims and stride == coeff and size == var_len:
                    best_input_dim = in_dim
                    break

            if best_input_dim is None:
                return None

            output_to_input.append(best_input_dim)
            used_input_dims.add(best_input_dim)

        # Check if we have a valid permutation
        if len(set(output_to_input)) != ndim:
            return None

        # Determine if iteration variables are in reversed order compared to output dimensions.
        # When var_lengths matches buf_sizes exactly, variables are in buffer dimension order.
        # When var_lengths is reversed from output shape, variables are innermost-to-outermost.
        #
        # For contiguous access (no transpose): var_lengths == buf_sizes in order, so
        #   output_to_input will be identity [0, 1, ...] - return None
        # For transpose: var_lengths are in reversed output order, so we need to
        #   reverse output_to_input to get actual axes.
        #
        # Check if output_to_input is identity (no transpose needed)
        identity_perm = list(range(ndim))
        if output_to_input == identity_perm:
            return None

        # Non-identity permutation means we need a transpose.
        # Determine actual axes based on var_lengths ordering.
        reversed_output = list(reversed(output_to_input))

        # Determine the correct axes for the transpose.
        # If var_lengths are in reversed order compared to a typical output shape,
        # we need to use reversed_output. Otherwise, use output_to_input directly.
        #
        # Check if var_lengths appears to be reversed (larger dims first for typical N-D arrays
        # where inner dims are usually smaller in memory-contiguous layouts)
        # For example, [16, 16, 4, 2] suggests reversed from [2, 4, 16, 16]

        # Use reversed axes if var_lengths looks like it's in innermost-first order
        # This is a heuristic: if the first var length is smaller or equal to the last
        # for the input buffer, we're likely in normal order; otherwise reversed
        if var_lengths[0] <= var_lengths[-1]:
            # Variables likely in normal order (outermost to innermost)
            actual_axes = output_to_input
        else:
            # Variables likely in reversed order (innermost to outermost)
            actual_axes = reversed_output

        # Check if actual_axes is identity (no transpose needed)
        # We checked output_to_input above, but actual_axes could be reversed_output
        # which might be identity even if output_to_input is not
        if list(actual_axes) == identity_perm:
            return None

        # The axes for jnp.transpose: axes[i] = which input dim goes to output dim i
        return tuple(actual_axes)

    def _is_transposed_access(self, name: str, index: sympy.Expr) -> bool:
        """Check if buffer access needs transpose (backward compatible wrapper)."""
        return self._get_transpose_axes(name, index) is not None

    def _get_noncontiguous_transpose_axes(self, name: str) -> Optional[tuple[int, ...]]:
        """
        Compute transpose axes needed for non-contiguous buffer inputs.

        When a non-contiguous tensor (e.g., from a transpose view) is passed to
        the kernel, JAX receives the data in its physical memory layout, not the
        logical shape. We need to apply a transpose to reorder the data correctly.

        Returns the axes permutation if the buffer is non-contiguous, None otherwise.
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return None

        buf_size = buf_obj.get_size()
        ndim = len(buf_size)

        # Need at least 2D for transpose
        if ndim < 2:
            return None

        layout = getattr(buf_obj, "get_layout", lambda: None)()
        if layout is None:
            return None

        buf_stride = getattr(layout, "stride", None)
        if buf_stride is None or len(buf_stride) != ndim:
            return None

        # Get buffer strides and sizes as integers
        buf_strides = [self._safe_int(s) for s in buf_stride]
        buf_sizes = [self._safe_int(s) for s in buf_size]
        if None in buf_strides or None in buf_sizes:
            return None

        # Skip if any dimension has size <= 1 (no meaningful transpose)
        if any(s <= 1 for s in buf_sizes):
            return None

        # Compute expected contiguous strides (row-major)
        expected_strides = []
        stride = 1
        for i in range(ndim - 1, -1, -1):
            expected_strides.insert(0, stride)
            stride *= buf_sizes[i]

        # Check if buffer is already contiguous
        if buf_strides == expected_strides:
            return None

        # Buffer is non-contiguous. Compute the permutation needed.
        # The physical layout is determined by strides: larger stride = outer dimension.
        # We need to find the permutation that maps physical order to logical order.

        # Create list of (stride, dim_index) and sort by stride descending
        # This gives us the physical dimension order (outermost to innermost)
        stride_dim_pairs = [(buf_strides[i], i) for i in range(ndim)]
        stride_dim_pairs.sort(key=lambda x: -x[0])  # Sort by stride, largest first

        # Build the inverse permutation: phys_to_logical[phys_dim] = logical_dim
        phys_to_logical = [dim for _, dim in stride_dim_pairs]

        # Check if this is just identity (already in correct order)
        if phys_to_logical == list(range(ndim)):
            return None

        # We need the inverse: for each logical dim, which physical dim provides it
        # axes[logical_dim] = physical_dim means: output dim logical_dim comes from
        # input (physical) dim physical_dim
        logical_to_phys = [0] * ndim
        for phys_dim, logical_dim in enumerate(phys_to_logical):
            logical_to_phys[logical_dim] = phys_dim

        return tuple(logical_to_phys)

    def _get_permute_axes_for_store(self) -> Optional[tuple[int, ...]]:
        """
        Compute permutation axes needed when input and output shapes differ.

        When a kernel has a permute/transpose operation, the input shape differs
        from the output shape. For example:
        - Input shape: (2, 16, 4, 16)
        - Output shape: (2, 16, 16, 4)
        - Permutation: [0, 1, 3, 2] (swap dims 2 and 3)

        This function detects such cases and returns the permutation axes.
        Returns None if no permutation is needed (shapes match or can't determine).
        """
        # Get the first input buffer shape
        input_shape = None
        for buf_name in self.args.input_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                input_shape = [self._safe_int(s) for s in buf_obj.get_size()]
                if None not in input_shape:
                    break
                input_shape = None

        if input_shape is None:
            return None

        # Get the first output buffer shape
        output_shape = None
        output_buffers = getattr(self.args, "output_buffers", {})
        for buf_name in output_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                output_shape = [self._safe_int(s) for s in buf_obj.get_size()]
                if None not in output_shape:
                    break
                output_shape = None

        if output_shape is None:
            return None

        # If shapes are the same, check if strides indicate a transpose is needed
        # This handles square matrices where shapes are equal but strides differ
        # (e.g., input strides [8, 1] vs output strides [1, 8])
        # BUT: for "copy" kernels where output strides are just a different layout
        # (not a true transpose/reverse), don't apply permutation - the .copy_()
        # in the main function will handle the layout conversion correctly.
        if input_shape == output_shape:
            # Get both input and output strides to determine if this is a true transpose
            input_strides = None
            output_strides = None
            for buf_name in self.args.input_buffers:
                buf_obj = V.graph.get_buffer(buf_name)
                if buf_obj is not None:
                    layout = getattr(buf_obj, "get_layout", lambda: None)()
                    if layout is not None:
                        stride = getattr(layout, "stride", None)
                        if stride is not None:
                            input_strides = [self._safe_int(s) for s in stride]
                            if None in input_strides:
                                input_strides = None
                            break
            output_buffers = getattr(self.args, "output_buffers", {})
            for buf_name in output_buffers:
                buf_obj = V.graph.get_buffer(buf_name)
                if buf_obj is not None:
                    layout = getattr(buf_obj, "get_layout", lambda: None)()
                    if layout is not None:
                        stride = getattr(layout, "stride", None)
                        if stride is not None:
                            output_strides = [self._safe_int(s) for s in stride]
                            if None in output_strides:
                                output_strides = None
                            break

            if input_strides is not None and output_strides is not None:
                # Check if output strides are the REVERSE of input strides
                # This indicates a true transpose (like x.T on square matrix)
                if output_strides == input_strides[::-1]:
                    # True transpose: apply permutation
                    return self._compute_permute_from_strides()
                # Otherwise, this is just a layout change (copy kernel)
                # Don't apply permutation - .copy_() handles layout conversion
                return None
            return self._compute_permute_from_strides()

        # If different number of dimensions, can't be a simple permutation
        if len(input_shape) != len(output_shape):
            return None

        # Check if output_shape is a permutation of input_shape
        if sorted(input_shape) != sorted(output_shape):
            return None  # Not a permutation (different elements)

        # IMPORTANT: If both input and output are contiguous (row-major strides),
        # this is a VIEW/reshape operation, NOT a transpose. A view just reinterprets
        # the same flat data with a different shape - no data reordering needed.
        # For a true transpose, the output would have non-contiguous strides.
        def is_contiguous_for_shape(shape, strides):
            """Check if strides are standard row-major for the given shape."""
            if strides is None or shape is None:
                return False
            expected_stride = 1
            for dim_size, actual_stride in reversed(list(zip(shape, strides))):
                if actual_stride != expected_stride:
                    return False
                expected_stride *= dim_size
            return True

        # Get output strides
        output_strides = None
        output_buffers = getattr(self.args, "output_buffers", {})
        for buf_name in output_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                layout = getattr(buf_obj, "get_layout", lambda: None)()
                if layout is not None:
                    stride = getattr(layout, "stride", None)
                    if stride is not None:
                        output_strides = [self._safe_int(s) for s in stride]
                        if None in output_strides:
                            output_strides = None
                        break

        # If output has contiguous strides for its shape, this is a view/reshape
        if is_contiguous_for_shape(output_shape, output_strides):
            return None  # No transpose needed - it's a view operation

        # Find the permutation: for each output dim, which input dim has the same size
        # This only works reliably when all dimensions have unique sizes
        if len(set(input_shape)) != len(input_shape):
            # Multiple dims have same size - need to use stride info instead
            # Try to use stride patterns to determine the permutation
            return self._compute_permute_from_strides()

        # All dims unique - can determine permutation from shapes
        axes = []
        used = set()
        for out_size in output_shape:
            for i, in_size in enumerate(input_shape):
                if in_size == out_size and i not in used:
                    axes.append(i)
                    used.add(i)
                    break

        if len(axes) != len(output_shape):
            return None

        # Check if it's identity (no permutation needed)
        if axes == list(range(len(axes))):
            return None

        return tuple(axes)

    def _get_permute_info_for_store(
        self,
    ) -> Optional[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        """
        Get permutation info for store when input/output shapes differ by permutation.

        Returns (input_shape, output_shape, permute_axes) or None if no permutation needed.
        This is used for the scatter store path where we need to reshape and transpose.
        """
        # Get the first input buffer shape
        input_shape = None
        for buf_name in self.args.input_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                shape = [self._safe_int(s) for s in buf_obj.get_size()]
                if None not in shape:
                    input_shape = tuple(shape)
                    break

        if input_shape is None:
            return None

        # Get the first output buffer shape
        output_shape = None
        output_buffers = getattr(self.args, "output_buffers", {})
        for buf_name in output_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                shape = [self._safe_int(s) for s in buf_obj.get_size()]
                if None not in shape:
                    output_shape = tuple(shape)
                    break

        if output_shape is None:
            return None

        # Check if they have same number of elements
        input_numel = 1
        for s in input_shape:
            input_numel *= s
        output_numel = 1
        for s in output_shape:
            output_numel *= s

        if input_numel != output_numel:
            return None

        # If shapes are the same, no permutation needed
        if input_shape == output_shape:
            return None

        # For different number of dims, need to find intermediate shape
        # Common case: input (2, 16, 4, 8, 2) -> output (2, 4, 16, 16)
        # Intermediate: (2, 16, 4, 16) with permute [0, 2, 1, 3]
        if len(input_shape) != len(output_shape):
            # Try to find intermediate shape by matching dimensions
            # The intermediate shape should have same ndim as output
            # and same numel as input
            intermediate = self._find_intermediate_shape_for_permute(
                input_shape, output_shape
            )
            if intermediate is None:
                return None
            input_shape = intermediate

        # Now input_shape and output_shape have same ndim
        # Check if output_shape is a permutation of input_shape
        if sorted(input_shape) != sorted(output_shape):
            return None

        # Find the permutation axes
        # For each output position, find which input position it comes from
        axes = []
        used = set()

        # First, handle dims with unique sizes - these can be matched unambiguously
        unique_input_sizes = {}
        for i, s in enumerate(input_shape):
            if input_shape.count(s) == 1:
                unique_input_sizes[s] = i

        for out_idx, out_size in enumerate(output_shape):
            if out_size in unique_input_sizes:
                # This size is unique in input, so we know the mapping
                in_idx = unique_input_sizes[out_size]
                if in_idx not in used:
                    axes.append(in_idx)
                    used.add(in_idx)
                    continue

            # For non-unique sizes, prefer matching same position first (identity)
            # Then try nearby positions
            matched = False
            # Prefer identity mapping (same position)
            if out_idx < len(input_shape) and input_shape[out_idx] == out_size and out_idx not in used:
                axes.append(out_idx)
                used.add(out_idx)
                matched = True
            else:
                # Try other positions with same size
                for in_idx, in_size in enumerate(input_shape):
                    if in_size == out_size and in_idx not in used:
                        axes.append(in_idx)
                        used.add(in_idx)
                        matched = True
                        break

            if not matched:
                return None

        if len(axes) != len(output_shape):
            return None

        # Check if it's identity (no permutation needed)
        if axes == list(range(len(axes))):
            return None

        return (input_shape, output_shape, tuple(axes))

    def _find_intermediate_shape_for_permute(
        self, input_shape: tuple[int, ...], output_shape: tuple[int, ...]
    ) -> Optional[tuple[int, ...]]:
        """
        Find intermediate shape when input and output have different ndim.

        For example:
        - Input (2, 16, 4, 8, 2) -> Output (2, 4, 16, 16)
        - Intermediate should be (2, 16, 4, 16) which is a view of input
          and can be permuted to output
        """
        # Common case: input has more dims, need to merge some dims
        # The merged dims typically come from view_as_real -> view pattern

        # Try to find which input dims to merge
        # Start from the end (common pattern: last N dims merge)
        for merge_start in range(len(input_shape) - 1, 0, -1):
            # Merge dims from merge_start to end
            merged_size = 1
            for i in range(merge_start, len(input_shape)):
                merged_size *= input_shape[i]

            # Build intermediate shape
            intermediate = list(input_shape[:merge_start]) + [merged_size]

            # Check if this matches output ndim and sorted values
            if len(intermediate) == len(output_shape):
                if sorted(intermediate) == sorted(output_shape):
                    return tuple(intermediate)

        return None

    def _compute_permute_from_strides(self) -> Optional[tuple[int, ...]]:
        """
        Compute permutation from stride patterns when sizes are not unique.
        """
        # Get input buffer stride info
        input_strides = None
        for buf_name in self.args.input_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                layout = getattr(buf_obj, "get_layout", lambda: None)()
                if layout is not None:
                    stride = getattr(layout, "stride", None)
                    if stride is not None:
                        input_strides = [self._safe_int(s) for s in stride]
                        if None not in input_strides:
                            break
                        input_strides = None

        # Get output buffer stride info
        output_strides = None
        output_buffers = getattr(self.args, "output_buffers", {})
        for buf_name in output_buffers:
            buf_obj = V.graph.get_buffer(buf_name)
            if buf_obj is not None:
                layout = getattr(buf_obj, "get_layout", lambda: None)()
                if layout is not None:
                    stride = getattr(layout, "stride", None)
                    if stride is not None:
                        output_strides = [self._safe_int(s) for s in stride]
                        if None not in output_strides:
                            break
                        output_strides = None

        if input_strides is None or output_strides is None:
            return None

        if len(input_strides) != len(output_strides):
            return None

        # Match strides: for each output stride, find matching input stride
        axes = []
        used = set()
        for out_stride in output_strides:
            found = False
            for i, in_stride in enumerate(input_strides):
                if in_stride == out_stride and i not in used:
                    axes.append(i)
                    used.add(i)
                    found = True
                    break
            if not found:
                return None

        if len(axes) != len(output_strides):
            return None

        # Check if it's identity
        if axes == list(range(len(axes))):
            return None

        return tuple(axes)

    def _has_column_major_output(self) -> bool:
        """Check if any output buffer has column-major stride layout."""
        output_buffers = getattr(self.args, "output_buffers", {})
        for buf_name in output_buffers:
            out_buf = V.graph.get_buffer(buf_name)
            if out_buf is None:
                continue
            layout = getattr(out_buf, "get_layout", lambda: None)()
            if layout is None:
                continue
            out_stride = getattr(layout, "stride", None)
            if out_stride is None or len(out_stride) < 2:
                continue
            out_s0 = self._safe_int(out_stride[0])
            out_s1 = self._safe_int(out_stride[1])
            if out_s0 is not None and out_s1 is not None and out_s0 < out_s1:
                return True

        # Also check graph buffers (output_buffers may not be populated during load)
        for buf_name in V.graph.name_to_buffer:
            out_buf = V.graph.get_buffer(buf_name)
            if out_buf is None or not isinstance(out_buf, ComputedBuffer):
                continue
            layout = getattr(out_buf, "get_layout", lambda: None)()
            if layout is None:
                continue
            out_stride = getattr(layout, "stride", None)
            if out_stride is None or len(out_stride) < 2:
                continue
            out_s0 = self._safe_int(out_stride[0])
            out_s1 = self._safe_int(out_stride[1])
            if out_s0 is not None and out_s1 is not None and out_s0 < out_s1:
                return True

        return False

    def _get_index_expr(self, index: sympy.Expr) -> tuple[str, bool]:
        """Get the index expression string and whether it needs flattening."""
        has_indirect = self._has_indirect_vars(index)
        has_iter_vars = self._has_iteration_vars(index)

        if has_indirect and has_iter_vars:
            return self._handle_mixed_indexing(index), True
        elif has_indirect:
            return self.kexpr(index), False
        else:
            index_str = self._get_index_str(index)
            # Check if index contains ModularIndexing - this requires flattened access
            # ModularIndexing is used for roll/wrap-around operations
            needs_flatten = index.has(ModularIndexing) and index_str != "..."
            # If index_str is an actual expression (not "..." or a slice pattern),
            # we need flattened access because it uses block variables
            if not needs_flatten and index_str != "...":
                # Check if it's a simple slice pattern (::N or M::N)
                if not ("::" in index_str or index_str.lstrip("-").isdigit()):
                    needs_flatten = True
            return index_str, needs_flatten

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        """Convert value to int, returning None on failure."""
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _compute_prefix_numel(self, prefixes: OrderedSet) -> Optional[int]:
        """Compute total numel for given prefixes (e.g., pointwise prefixes)."""
        result = 1
        for p in prefixes:
            if p in self.numels:
                numel = self._safe_int(self.numels[p])
                if numel is None:
                    return None
                result *= numel
        return result

    def _compute_reduction_numel(self) -> Optional[int]:
        """Compute total reduction numel."""
        result = 1
        for tree in self.range_trees:
            if tree.is_reduction:
                numel = self._safe_int(tree.numel)
                if numel is None:
                    return None
                result *= numel
        return result

    def _can_use_tma_approach(self) -> bool:
        """
        Check if TMA (Tensor Memory Accelerator) approach can be used.
        TMA works for simple element-wise ops but not for:
        - Reductions (need different accumulation patterns)
          TODO: TMA supports float64 for loading but not for reductions
        - Broadcasting (inputs have different shapes or output differs)
        - Non-contiguous tensors (strided, transposed)
        """
        # Check for reductions
        reduction_numel = self._compute_reduction_numel()
        if reduction_numel is not None and reduction_numel > 1:
            return False

        # Check all input buffers for contiguity, dtype, and shape consistency
        input_shapes: list[tuple] = []
        for name in self.args.input_buffers:
            buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = (
                self._get_buffer_info(name)
            )
            if not is_contiguous:
                return False

            # Check for unsupported dtypes
            # TODO: TMA supports float64 for loading but current JAX Mosaic GPU
            # implementation doesn't support it yet. Re-enable when JAX adds support.
            buf_dtype = getattr(buf_obj, "get_dtype", lambda: None)()
            if buf_dtype is not None:
                import torch

                if buf_dtype == torch.float64:
                    return False

            # Collect shape as tuple for comparison
            shape_tuple = tuple(self._safe_int(s) for s in buf_size)
            if None in shape_tuple:
                return False  # Dynamic shapes not supported
            input_shapes.append(shape_tuple)

        # Check if all input shapes are identical (no broadcasting)
        if input_shapes and len(OrderedSet(input_shapes)) > 1:
            return False

        # Check that output numel matches input numel (no broadcasting expansion)
        if input_shapes:
            input_numel = 1
            for s in input_shapes[0]:
                input_numel *= s

            # Compute output numel from pointwise range trees (non-reduction)
            output_numel = 1
            for tree in self.range_trees:
                if not tree.is_reduction:
                    numel = self._safe_int(tree.numel)
                    if numel is None:
                        return False  # Dynamic shapes not supported
                    output_numel *= numel

            if output_numel != input_numel:
                return False

        return True

    def _get_buffer_info(self, name: str) -> tuple[Any, Any, Any, list, bool]:
        """Get buffer metadata (buf_obj, buf_size, buf_numel, actual_strides, is_contiguous)."""
        buf_obj = V.graph.get_buffer(name)
        buf_size = buf_obj.get_size()
        buf_numel = 1
        for s in buf_size:
            sval = self._safe_int(s)
            buf_numel *= sval if sval is not None else s

        # Get buffer strides and check contiguity
        actual_strides: list = []
        is_contiguous = True

        layout = getattr(buf_obj, "get_layout", lambda: None)()
        buf_stride = getattr(layout, "stride", None) if layout else None

        if buf_stride is not None:
            for i in range(len(buf_size)):
                actual_stride = self._safe_int(buf_stride[i])
                actual_strides.append(actual_stride)

            # Check contiguity
            if len(buf_size) == 1:
                if actual_strides[0] is not None and actual_strides[0] != 1:
                    is_contiguous = False
            elif len(buf_size) > 1:
                expected_stride = 1
                for i in range(len(buf_size) - 1, -1, -1):
                    actual_stride = actual_strides[i]
                    if actual_stride is None or actual_stride != expected_stride:
                        is_contiguous = False
                    dim_size = self._safe_int(buf_size[i])
                    if dim_size is not None:
                        expected_stride *= dim_size

        return buf_obj, buf_size, buf_numel, actual_strides, is_contiguous

    def _compute_output_numel_from_index(
        self, index: sympy.Expr
    ) -> tuple[int, OrderedSet]:
        """Compute expected output numel and used vars from iteration variables in index."""
        used_vars = self._get_used_iter_vars(index)

        used_range_lengths = []
        for var in used_vars:
            if var in self.range_tree_nodes:
                entry = self.range_tree_nodes[var]
                length_val = self._safe_int(entry.length)
                if length_val is not None:
                    used_range_lengths.append(length_val)

        output_numel = 1
        for l in used_range_lengths:
            output_numel *= l

        return output_numel, used_vars

    def _get_index_coefficients(
        self, index: sympy.Expr, used_vars: OrderedSet
    ) -> OrderedSet:
        """
        Extract coefficients of iteration variables from index expression.
        """
        coefficients: OrderedSet = OrderedSet()
        for var in used_vars:
            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)
            stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
            if stride is None:
                stride = 1  # Variable without explicit coefficient has stride 1
            coef = self._safe_int(stride)
            coefficients.add(coef if coef is not None else stride)
        return coefficients

    def _check_gather_pattern(
        self,
        buf_size: list,
        actual_strides: list,
        is_contiguous: bool,
        coefficients: OrderedSet,
    ) -> bool:
        """
        Check if access pattern requires gather (non-standard striding).
        """
        expected_strides = [1]  # 1D buffers have stride 1

        if len(buf_size) > 1:
            expected_stride = 1
            expected_strides = []
            for i in range(len(buf_size) - 1, -1, -1):
                expected_strides.insert(0, expected_stride)
                dim_size = self._safe_int(buf_size[i])
                if dim_size is not None:
                    expected_stride *= dim_size

        if is_contiguous:
            # Buffer is contiguous - check if access coefficients match expected strides
            expected_stride_set = OrderedSet(expected_strides)
            for coef in coefficients:
                if coef not in expected_stride_set:
                    return True
        else:
            # Buffer is NOT contiguous (strided input)
            # Check if coefficients match actual buffer strides
            actual_stride_set = OrderedSet(s for s in actual_strides if s is not None)
            for coef in coefficients:
                if coef not in actual_stride_set:
                    return True

        return False

    def _needs_strided_indexing(
        self,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> tuple[str, bool]:
        """
        Check if buffer access needs strided indexing due to size mismatch or gather patterns.

        This handles cases like:
        - Pooling operations where input/output have different sizes
        - im2col-like gather patterns
        - Transposed or strided buffer access
        """
        # Only applies when full array access is indicated
        if index_str != "..." or needs_flatten:
            return index_str, needs_flatten

        buf = V.graph.get_buffer(name)
        if buf is None:
            return index_str, needs_flatten

        buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = (
            self._get_buffer_info(name)
        )
        output_numel, used_vars = self._compute_output_numel_from_index(index)
        all_iter_vars = self._get_iter_vars()
        coefficients = self._get_index_coefficients(index, used_vars)

        # Check for gather pattern
        has_non_unit_strides = self._check_gather_pattern(
            buf_size, actual_strides, is_contiguous, coefficients
        )

        # Check for im2col-like pattern (more iter vars used than buffer dims)
        buf_effective_dims = sum(1 for s in buf_size if self._safe_int(s) != 1)
        not_all_vars_used = (
            len(used_vars) < len(all_iter_vars)
            and len(used_vars) > 0
            and buf_effective_dims > 1
            and len(used_vars) > len(buf_size)
        )

        # Check various conditions for skipping strided indexing
        is_tpu = torch._inductor.config._debug_cpu_to_tpu_pallas
        is_known_non_contiguous = not is_contiguous and all(
            s is not None for s in actual_strides
        )
        has_symbolic_coef = any(not isinstance(c, int | float) for c in coefficients)
        skip_for_non_contiguous = (
            is_known_non_contiguous and not is_tpu and buf_numel == output_numel
        )

        # Determine if strided indexing is needed
        if (
            output_numel > 0
            and (buf_numel != output_numel or not_all_vars_used or has_non_unit_strides)
            and len(used_vars) > 0
            and not skip_for_non_contiguous
            and not has_symbolic_coef
        ):
            return self._generate_strided_index(index), True

        return index_str, needs_flatten

    def _adjust_index_for_buffer_shape(
        self,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> tuple[str, bool]:
        """
        Adjust index expression based on buffer shape (0-dim scalar, multi-dim, etc.).
        """
        if needs_flatten or index_str == "...":
            return index_str, needs_flatten

        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return index_str, needs_flatten

        buf_size = buf_obj.get_size()

        # 0-dimensional (scalar) buffer - use [...] to access it
        if len(buf_size) == 0:
            return "...", needs_flatten

        # Multi-dimensional buffer with constant/scalar index
        if len(buf_size) > 1:
            has_iter_vars = self._has_iteration_vars(index)
            if not has_iter_vars:
                return index_str, True  # Use flattened access
            elif "::" in index_str:
                # Strided slice patterns need flattened indexing for multi-dim
                return self._generate_strided_index(index), True

        # GPU doesn't support gather from slice patterns on 1D buffers
        if self.is_gpu and "::" in index_str:
            return self._generate_strided_index(index), True

        return index_str, needs_flatten

    def _build_load_expr(
        self,
        buf: str,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> str:
        """
        Build the load expression based on indexing mode.
        """

        if needs_flatten:
            # Flatten then index for non-contiguous access (gather operation)
            has_minmax = index.has(sympy.Min) or index.has(sympy.Max)
            idx = f"({index_str}).astype(jnp.int64)" if has_minmax else index_str
            load_expr = f"{buf}[...].flatten()[{idx}]"

            # For indexed loads, the result shape is determined by index expression
            # broadcast. This may not match the canonical output shape needed for
            # correct binary op broadcasting. Reshape to canonical shape when:
            # 1. We have a canonical output shape
            # 2. Buffer numel matches canonical numel (same total elements)
            # This ensures shapes like (6, 2) become (2, 3, 2) for correct broadcasting.
            canonical = self._get_canonical_output_shape()
            if canonical is not None:
                canonical_numel = 1
                for s in canonical:
                    canonical_numel *= s

                # Try to get buffer size to verify numel matches
                buf_obj = V.graph.get_buffer(name)
                should_reshape = False
                if buf_obj is not None:
                    buf_size = buf_obj.get_size()
                    buf_numel = 1
                    for s in buf_size:
                        buf_numel_val = self._safe_int(s)
                        if buf_numel_val is None:
                            buf_numel = None
                            break
                        buf_numel *= buf_numel_val

                    if buf_numel is not None and buf_numel == canonical_numel:
                        should_reshape = True
                else:
                    # For input buffers not in graph.buffers, reshape unconditionally
                    # since indexed loads must produce matching numel
                    should_reshape = True

                if should_reshape:
                    shape_str = ", ".join(str(s) for s in canonical)
                    load_expr = f"{load_expr}.reshape({shape_str})"

            return load_expr
        else:
            # Direct indexing for contiguous access
            load_expr = f"{buf}[{index_str}]"

            # Check for transposed access (N-D transpose support)
            if index_str == "...":
                transpose_axes = self._get_transpose_axes(name, index)
                if transpose_axes is not None:
                    axes_str = ", ".join(str(a) for a in transpose_axes)
                    load_expr = f"jnp.transpose({load_expr}, axes=({axes_str},))"
                    self.has_transposed_load = True
                else:
                    # Also check if buffer shape is transpose of output shape
                    # This handles cases where index pattern is normal but output
                    # shape differs (e.g., x.t() where x is accessed normally but
                    # output needs transposed shape)
                    load_expr = self._maybe_transpose_for_output_shape(
                        name, load_expr
                    )

            return load_expr

    def _maybe_transpose_for_output_shape(
        self, name: str, load_expr: str
    ) -> str:
        """
        Transpose buffer if its shape is the transpose of the output shape.

        This handles cases like x.t() + y where:
        - x has shape (N, M) with normal row-major strides
        - x.t() should have shape (M, N)
        - The kernel output shape is (M, N)

        The index expression accesses x in normal row-major order, but we need
        the data in transposed shape for the output. In this case, we need to
        apply jnp.transpose() on load.
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return load_expr

        buf_size = buf_obj.get_size()
        buf_size_ints = [self._safe_int(s) for s in buf_size]
        if None in buf_size_ints or len(buf_size_ints) != 2:
            return load_expr

        # Get output buffer shape
        output_shape = None
        output_buffers = getattr(self.args, "output_buffers", {})
        for out_name in output_buffers:
            out_buf = V.graph.get_buffer(out_name)
            if out_buf is not None:
                out_size = [self._safe_int(s) for s in out_buf.get_size()]
                if None not in out_size and len(out_size) == 2:
                    output_shape = out_size
                    break

        if output_shape is None:
            return load_expr

        # Check if buffer shape is transpose of output shape (same dims, reversed)
        if (
            buf_size_ints != output_shape
            and sorted(buf_size_ints) == sorted(output_shape)
            and buf_size_ints == list(reversed(output_shape))
        ):
            self.has_transposed_load = True
            return f"jnp.transpose({load_expr})"

        return load_expr

    def _maybe_squeeze_intermediate_buffer(self, name: str, load_expr: str) -> str:
        """
        Squeeze (N,1) intermediate buffers when kernel has 1D graph inputs.

        This avoids wrong broadcasting: (N,) op (N,1) -> (N,N) instead of (N,)
        """
        if not name.startswith("buf"):
            return load_expr

        # Check if any input buffer is a 1D graph input
        has_1d_input = any(
            not buf_name.startswith("buf")
            and (buf_obj := V.graph.get_buffer(buf_name)) is not None
            and len(buf_obj.get_size()) == 1
            for buf_name in self.args.input_buffers
        )

        if has_1d_input:
            buf_obj = V.graph.get_buffer(name)
            if buf_obj is not None:
                buf_size = buf_obj.get_size()
                if len(buf_size) == 2 and buf_size[-1] == 1:
                    return f"jnp.squeeze({load_expr}, axis=-1)"

        return load_expr

    def _maybe_reshape_intermediate_buffer(self, name: str, load_expr: str) -> str:
        """
        Reshape buffers to match the canonical output shape.

        This handles cases like embedding + residual where:
        - Embedding output has shape (batch, seq, hidden) = (2, 16, 64)
        - Intermediate buffer has shape (batch*seq, hidden) = (32, 64)
        These need reshape to broadcast correctly.
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return load_expr

        # Don't reshape non-floating-point tensors (like int64 indices)
        # They have specific shapes that shouldn't be changed for broadcasting
        dtype = V.graph.get_dtype(name)
        if dtype is not None and not dtype.is_floating_point:
            return load_expr

        buf_size = buf_obj.get_size()
        buf_size_ints = [self._safe_int(s) for s in buf_size]
        if None in buf_size_ints:
            return load_expr

        # Get canonical output shape for this kernel
        output_shape = self._get_canonical_output_shape()
        if output_shape is None:
            return load_expr

        # Compute numels for different shape options
        buf_numel = 1
        for s in buf_size_ints:
            buf_numel *= s
        output_numel = 1
        for s in output_shape:
            output_numel *= s
        output_shape_no_keepdims = [s for s in output_shape if s != 1]
        output_numel_no_keepdims = 1
        for s in output_shape_no_keepdims:
            output_numel_no_keepdims *= s

        # Check if buffer already matches target shape
        if buf_size_ints == list(output_shape):
            return load_expr
        if buf_size_ints == output_shape_no_keepdims:
            return load_expr

        # Check if buffer shape is a TRANSPOSE of output shape (same dims, different order)
        # In this case, we need jnp.transpose(), not reshape() - they have different semantics!
        # reshape() just reinterprets the flat array; transpose() actually reorders elements.
        if len(buf_size_ints) == len(output_shape) and sorted(buf_size_ints) == sorted(output_shape):
            # Check if it's a simple 2D transpose (most common case)
            if len(buf_size_ints) == 2 and buf_size_ints == list(reversed(output_shape)):
                self.has_transposed_load = True
                return f"jnp.transpose({load_expr})"
            # For N-D, compute the permutation
            if len(set(buf_size_ints)) == len(buf_size_ints):  # All dims unique
                axes = []
                for out_dim_size in output_shape:
                    for i, buf_dim_size in enumerate(buf_size_ints):
                        if buf_dim_size == out_dim_size and i not in axes:
                            axes.append(i)
                            break
                if len(axes) == len(output_shape) and axes != list(range(len(axes))):
                    axes_str = ", ".join(str(a) for a in axes)
                    self.has_transposed_load = True
                    return f"jnp.transpose({load_expr}, axes=({axes_str},))"

        # Decide which target shape to use based on buffer characteristics:
        # - If buffer has a 1 dimension (likely keepdims result), use full output_shape
        # - If buffer numel matches full output shape, use full output_shape
        # - Otherwise, use output_shape without keepdims
        has_keepdims_dim = 1 in buf_size_ints
        if has_keepdims_dim and buf_numel == output_numel:
            # Buffer has keepdims dimension, reshape to full canonical shape
            if len(output_shape) > len(buf_size_ints):
                shape_str = ", ".join(str(s) for s in output_shape)
                return f"{load_expr}.reshape({shape_str})"
        elif buf_numel == output_numel_no_keepdims:
            # Buffer matches non-keepdims numel, reshape to non-keepdims shape
            if len(output_shape_no_keepdims) > len(buf_size_ints):
                shape_str = ", ".join(str(s) for s in output_shape_no_keepdims)
                return f"{load_expr}.reshape({shape_str})"
        elif buf_numel == output_numel:
            # Buffer matches full numel, reshape to full shape
            if len(output_shape) > len(buf_size_ints):
                shape_str = ", ".join(str(s) for s in output_shape)
                return f"{load_expr}.reshape({shape_str})"

        return load_expr

    def _maybe_broadcast_1d_buffer(
        self, name: str, index: sympy.Expr, load_expr: str
    ) -> str:
        """Reshape 1D buffers (e.g., batch norm mean) for higher-dim broadcasting."""
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None or len(buf_obj.get_size()) != 1:
            return load_expr

        buf_length = self._safe_int(buf_obj.get_size()[0])
        if buf_length is None:
            return load_expr

        # Only graph inputs, not intermediate buffers or index tensors
        if name.startswith("buf"):
            return load_expr
        dtype = V.graph.get_dtype(name)
        if dtype is not None and not dtype.is_floating_point:
            return load_expr

        # Find a higher-dimensional reference buffer
        ref_buf_size = None
        for buf_name in self.args.input_buffers:
            other_buf = V.graph.get_buffer(buf_name)
            if other_buf is not None and len(other_buf.get_size()) > 1:
                ref_buf_size = [self._safe_int(s) for s in other_buf.get_size()]
                if all(s is not None for s in ref_buf_size):
                    break
                ref_buf_size = None
        if ref_buf_size is None or len(ref_buf_size) <= 1:
            return load_expr

        # Must use exactly one iteration variable
        used_vars = self._get_used_iter_vars(index)
        if len(used_vars) != 1:
            return load_expr
        used_var = next(iter(used_vars))
        if used_var not in self.range_tree_nodes:
            return load_expr

        # Verify buffer length matches variable length
        entry = self.range_tree_nodes[used_var]
        if self._safe_int(entry.length) != buf_length:
            return load_expr

        # Buffer length must uniquely match one iteration variable
        matching_vars = [
            v
            for v, e in self.range_tree_nodes.items()
            if self._safe_int(e.length) == buf_length and not e.is_reduction
        ]
        if len(matching_vars) != 1:
            return load_expr

        # Buffer length must uniquely match one ref buffer dimension
        matching_dims = [i for i, s in enumerate(ref_buf_size) if s == buf_length]
        if len(matching_dims) != 1:
            return load_expr

        axis_pos = matching_dims[0]
        if axis_pos == len(ref_buf_size) - 1:
            return load_expr  # Last dim uses default broadcasting

        reshape_dims = [1] * len(ref_buf_size)
        reshape_dims[axis_pos] = -1
        return f"{load_expr}.reshape({', '.join(map(str, reshape_dims))})"

    def _maybe_reshape_for_expand(
        self, name: str, index: sympy.Expr, load_expr: str
    ) -> str:
        """
        Reshape buffer for expand patterns (stride=0 dimensions).

        When cloning an expanded view, the input buffer has fewer elements than
        the output. This detects which iteration variables are unused (coefficient=0
        in the load index, corresponding to stride=0 expanded dimensions) and
        reshapes the loaded buffer to insert singleton dimensions at those positions.

        For example, if output shape is (2, 16, 2, 2, 16) and input is (2, 16, 2, 16),
        and dimension 3 is expanded (stride=0), we reshape input from (2, 16, 2, 16)
        to (2, 16, 2, 1, 16) so JAX broadcast_to can expand to (2, 16, 2, 2, 16).
        """
        # Skip if load_expr already has a reshape (e.g., from _maybe_broadcast_1d_buffer)
        # to avoid double reshaping which causes shape mismatches
        if ".reshape(" in load_expr:
            return load_expr

        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return load_expr

        buf_size = buf_obj.get_size()
        buf_size_ints = [self._safe_int(s) for s in buf_size]
        if None in buf_size_ints:
            return load_expr

        buf_numel = 1
        for s in buf_size_ints:
            buf_numel *= s

        # Get all iteration variables (excluding reduction vars for output shape)
        all_iter_vars = list(self.range_tree_nodes.keys())
        used_vars = self._get_used_iter_vars(index)

        # Find unused iteration variables (these correspond to expanded dims)
        unused_vars = [v for v in all_iter_vars if v not in used_vars]
        if not unused_vars:
            return load_expr

        # Build output shape from iteration variables, sorted by divisor (high to low)
        # This gives dimension order: higher divisor = earlier dimension
        var_items = sorted(
            [
                (v, e)
                for v, e in self.range_tree_nodes.items()
                if not e.is_reduction
            ],
            key=lambda x: self._safe_int(x[1].divisor) or 0,
            reverse=True,
        )

        # Compute output numel from iteration variable lengths
        # Detect and exclude linear index variables (divisor=1, length=product of others)
        var_lengths: list[tuple[int | None, int | None]] = []
        for var, entry in var_items:
            length = self._safe_int(entry.length)
            divisor = self._safe_int(entry.divisor)
            var_lengths.append((length, divisor))

        # Identify linear index variables (divisor=1, length=product of others)
        # These are synthetic variables that shouldn't affect shape calculations
        linear_index_vars: set = set()
        for i, ((var, _), (length, divisor)) in enumerate(
            zip(var_items, var_lengths)
        ):
            if length is None:
                continue
            if divisor == 1 and len(var_lengths) > 1:
                other_product = 1
                for j, (other_len, _) in enumerate(var_lengths):
                    if j != i and other_len is not None:
                        other_product *= other_len
                if length == other_product:
                    linear_index_vars.add(var)

        output_numel = 1
        for i, (length, divisor) in enumerate(var_lengths):
            if length is None:
                continue
            var, _ = var_items[i]
            if var in linear_index_vars:
                continue
            output_numel *= length

        # Only handle expand pattern: output has more elements than input
        if buf_numel >= output_numel or buf_numel == 0:
            return load_expr

        # Build the target shape: use buffer dimensions for used vars, 1 for unused vars
        # The order is determined by divisor (higher divisor = earlier dimension)
        # Skip linear index variables as they don't correspond to actual dimensions
        target_shape = []
        buf_dim_idx = 0
        for var, entry in var_items:
            # Skip linear index variables - they don't represent real dimensions
            if var in linear_index_vars:
                continue
            if var in unused_vars:
                # This is an expanded dimension - insert singleton
                target_shape.append(1)
            else:
                # This dimension maps to actual buffer data
                if buf_dim_idx < len(buf_size_ints):
                    target_shape.append(buf_size_ints[buf_dim_idx])
                    buf_dim_idx += 1
                else:
                    # More used vars than buffer dims - fallback
                    return load_expr

        if buf_dim_idx != len(buf_size_ints):
            # Didn't use all buffer dimensions - fallback
            return load_expr

        # Check if reshape is actually needed
        if target_shape == buf_size_ints:
            return load_expr

        shape_str = ", ".join(str(s) for s in target_shape)
        return f"{load_expr}.reshape({shape_str})"

    def _check_im2col_pattern(
        self, index: sympy.Expr, index_str: str, needs_flatten: bool
    ) -> tuple[str, bool]:
        """
        Check for im2col-like patterns where store uses block variables but load doesn't.

        For cat/expand patterns, both load and store prepared indices share block vars.
        For im2col patterns, store compresses to block vars but load doesn't.
        """
        if index_str != "..." or needs_flatten:
            return index_str, needs_flatten

        prepared_index = self.prepare_indexing(index)
        iter_vars = self._get_iter_vars()
        store_orig_vars = self._get_used_iter_vars(index)
        store_prep_vars = (
            prepared_index.free_symbols
            if hasattr(prepared_index, "free_symbols")
            else OrderedSet()
        ) & iter_vars
        new_vars = store_prep_vars - store_orig_vars

        # Only trigger if store introduces new block vars
        if not new_vars or len(store_orig_vars) <= 1:
            return index_str, needs_flatten

        # Check if loads are compatible with broadcast or cat pattern
        has_im2col_pattern = False
        for buf_name, load_index in self.load_index_exprs.items():
            load_orig_vars = self._get_used_iter_vars(load_index)
            if not load_orig_vars:
                continue

            # Load has iteration variables
            if load_orig_vars != store_orig_vars:
                continue

            # Same vars - check if load gets compressed too
            prep_load = self.prepare_indexing(load_index)
            load_prep_vars = (
                prep_load.free_symbols
                if hasattr(prep_load, "free_symbols")
                else OrderedSet()
            ) & iter_vars

            # If store compresses but load doesn't, check for strided input vs im2col
            if load_orig_vars != load_prep_vars or store_prep_vars == store_orig_vars:
                continue

            # Check if load coefficients match buffer strides
            if not self._check_load_is_strided_input(
                buf_name, load_index, load_orig_vars
            ):
                has_im2col_pattern = True
                break

        if has_im2col_pattern:
            return self._generate_strided_index(prepared_index), True

        return index_str, needs_flatten

    def _check_load_is_strided_input(
        self, buf_name: str, load_index: sympy.Expr, load_orig_vars: OrderedSet
    ) -> bool:
        """
        Check if load coefficients match buffer strides (strided input vs im2col).
        """
        buf = V.graph.get_buffer(buf_name)
        if buf is None:
            return False

        layout = getattr(buf, "get_layout", lambda: None)()
        if layout is None:
            return False

        buf_strides = getattr(layout, "stride", None)
        if buf_strides is None:
            return False

        buf_sizes = buf.get_size()

        # Get load coefficients
        load_coeffs = []
        for var in load_orig_vars:
            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(load_index, var)
            coef = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
            if coef is not None:
                int_coef = self._safe_int(coef)
                load_coeffs.append(int_coef if int_coef is not None else coef)

        # Check if coefficients match buffer strides
        # Only include strides for non-trivial dimensions (size > 1)
        buf_stride_set = OrderedSet()
        for i, s in enumerate(buf_strides):
            dim_size = self._safe_int(buf_sizes[i])
            if dim_size is None or dim_size > 1:
                int_s = self._safe_int(s)
                buf_stride_set.add(int_s if int_s is not None else s)

        return OrderedSet(load_coeffs) == buf_stride_set

    def _check_store_needs_transpose(self, name: str) -> bool:
        """
        Check if output needs transpose for column-major storage.

        Transpose on store is needed when:
        - Output has column-major stride (s0 < s1)
        - But input(s) have row-major stride
        - And we haven't already transposed on load
        """
        if self.has_transposed_load:
            return False

        buf = V.graph.get_buffer(name)
        if buf is None:
            return False

        layout = getattr(buf, "get_layout", lambda: None)()
        if layout is None:
            return False

        buf_stride = getattr(layout, "stride", None)
        if buf_stride is None:
            return False

        buf_size = buf.get_size()
        if len(buf_stride) != 2 or len(buf_size) != 2:
            return False

        size0 = self._safe_int(buf_size[0])
        size1 = self._safe_int(buf_size[1])
        s0 = self._safe_int(buf_stride[0])
        s1 = self._safe_int(buf_stride[1])

        # Check if output is column-major with valid dimensions
        if not (
            s0 is not None
            and s1 is not None
            and s0 < s1
            and size0 is not None
            and size1 is not None
            and size0 > 1
            and size1 > 1
        ):
            return False

        # Check if any input is column-major (if so, no transpose needed)
        for inp_name in self.args.input_buffers:
            inp_buf = V.graph.get_buffer(inp_name)
            if inp_buf is None:
                continue
            inp_layout = getattr(inp_buf, "get_layout", lambda: None)()
            if inp_layout is None:
                continue
            inp_stride = getattr(inp_layout, "stride", None)
            if inp_stride is None or len(inp_stride) != 2:
                continue
            inp_s0 = self._safe_int(inp_stride[0])
            inp_s1 = self._safe_int(inp_stride[1])
            if inp_s0 is not None and inp_s1 is not None and inp_s0 < inp_s1:
                return False  # Input is also column-major

        return True

    def _get_broadcast_shape_for_expand(self, name: str) -> Optional[tuple[int, ...]]:
        """
        Get the intermediate broadcast shape for an expand pattern.

        For expand operations (stride=0 dimensions), returns the output shape
        with 1s at the expanded (stride=0) dimensions. This allows the value
        to be reshaped to this intermediate shape and then broadcast.

        Returns None if stride info is not available or if there are no
        stride=0 dimensions (not an expand pattern).
        """
        buf = V.graph.get_buffer(name)
        if buf is None:
            return None

        layout = getattr(buf, "get_layout", lambda: None)()
        if layout is None:
            return None

        buf_stride = getattr(layout, "stride", None)
        buf_size = buf.get_size()

        if buf_stride is None or len(buf_stride) != len(buf_size):
            return None

        # Build intermediate shape: use 1 where stride=0, original size elsewhere
        intermediate_shape = []
        has_expand = False
        for i, (size, stride) in enumerate(zip(buf_size, buf_stride)):
            size_int = self._safe_int(size)
            stride_int = self._safe_int(stride)
            if size_int is None:
                return None  # Dynamic size, can't precompute
            if stride_int == 0:
                intermediate_shape.append(1)
                has_expand = True
            else:
                intermediate_shape.append(size_int)

        # Only return shape if there's actually an expand (stride=0) dimension
        # Otherwise let the runtime helper figure it out
        if not has_expand:
            return None

        return tuple(intermediate_shape)

    def _build_full_array_store_expr(
        self, out: str, value: CSEVariable, needs_transpose: bool,
        broadcast_shape: Optional[tuple[int, ...]] = None,
        permute_axes: Optional[tuple[int, ...]] = None
    ) -> str:
        """
        Build store expression for full array assignment.

        Handles scalar broadcast, shape matching, optional transpose, and
        expand patterns (where input needs singleton dims inserted for broadcast).

        Args:
            broadcast_shape: If provided, the intermediate shape to reshape to
                           before broadcasting (for expand patterns with stride=0).
            permute_axes: If provided, apply jnp.transpose with these axes when
                         value shape differs from output shape (permute operation).
        """
        if needs_transpose:
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.transpose(jnp.asarray({value})))"
            )
        elif broadcast_shape is not None:
            # Use precomputed broadcast shape for expand pattern
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else (jnp.broadcast_to(jnp.asarray({value}).reshape(-1).reshape({broadcast_shape}), {out}.shape) "
                f"if jnp.asarray({value}).size != {out}.size "
                f"else jnp.asarray({value}).reshape({out}.shape)))"
            )
        elif permute_axes is not None:
            # Apply permutation when input and output shapes differ by a transpose
            axes_str = ", ".join(str(a) for a in permute_axes)
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.transpose(jnp.asarray({value}), axes=({axes_str},)))"
            )
        else:
            # Fall back to runtime helper for expand patterns without stride info.
            # Use _pallas_expand_for_broadcast helper which handles this at runtime.
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else (_pallas_expand_for_broadcast(jnp.asarray({value}), {out}.shape) "
                f"if jnp.asarray({value}).size != {out}.size "
                f"else jnp.asarray({value}).reshape({out}.shape)))"
            )

    def _build_store_expr(
        self,
        out: str,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        index_str: str,
        needs_flatten: bool,
        mode: Any = None,
    ) -> str:
        """
        Build the store expression based on indexing mode.
        mode can be None (set) or "atomic_add" (accumulate).
        """
        if index_str == "...":
            # Full array store with shape matching
            needs_transpose = self._check_store_needs_transpose(name)
            broadcast_shape = self._get_broadcast_shape_for_expand(name)
            # Check if a permutation is needed (input/output shapes differ by transpose)
            # IMPORTANT: Skip permute_axes if we already transposed on load - the data
            # is already in the correct shape and doesn't need another permutation.
            permute_axes = (
                self._get_permute_axes_for_store()
                if not needs_transpose and broadcast_shape is None and not self.has_transposed_load
                else None
            )
            return self._build_full_array_store_expr(out, value, needs_transpose, broadcast_shape, permute_axes)

        if needs_flatten:
            # Block variable indexing (e.g., im2col) - use flattened scatter
            scatter_op = "add" if mode == "atomic_add" else "set"
            # Check if a permutation is needed (input/output shapes differ by transpose)
            permute_info = self._get_permute_info_for_store()
            if permute_info is not None:
                input_shape, output_shape, permute_axes = permute_info
                input_shape_str = ", ".join(str(s) for s in input_shape)
                axes_str = ", ".join(str(a) for a in permute_axes)
                # Reshape value to intermediate shape, transpose, then flatten
                return (
                    f"{out}[...] = {out}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                    f"jnp.transpose(jnp.asarray({value}).reshape({input_shape_str}), "
                    f"axes=({axes_str},)).flatten()).reshape({out}.shape)"
                )
            return (
                f"{out}[...] = {out}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                f"jnp.asarray({value}).flatten()).reshape({out}.shape)"
            )

        # Direct indexed assignment
        has_indirect = self._has_indirect_vars(index)
        buf = V.graph.get_buffer(name)

        if buf is not None:
            buf_size = buf.get_size()
            if len(buf_size) > 1 and not self._has_iteration_vars(index):
                # Multi-dim output with constant index - use [...] for full assignment
                broadcast_shape = self._get_broadcast_shape_for_expand(name)
                return self._build_full_array_store_expr(out, value, False, broadcast_shape)

        if has_indirect:
            # Indirect indexed store (scatter): use .add() for atomic_add, .set() otherwise
            scatter_op = "add" if mode == "atomic_add" else "set"
            value_expr = (
                f"(jnp.full({index_str}.shape, {value}) "
                f"if jnp.asarray({value}).ndim == 0 else {value})"
            )
            if mode == "atomic_add":
                # For atomic_add, mark output as needing to be readable (for aliasing)
                self.outputs_need_read.add(out)
                alias_param = f"{out}_alias"
                return (
                    f"{out}[...] = {alias_param}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                    f"{value_expr}.flatten()).reshape({out}.shape)"
                )
            else:
                return f"{out}[{index_str}] = {value_expr}"

        return f"{out}[{index_str}] = {value}"

    def _build_scatter_store_expr(
        self,
        out: str,
        value: CSEVariable,
        scatter_info: dict[str, Any],
        name: str,
        mode: Any,
    ) -> str:
        """Build store expression for scatter operations (indirect indexing)."""
        is_point_scatter = scatter_info.get("is_point_scatter", False)

        # Mark this output parameter as needing to be readable (for aliasing)
        self.outputs_need_read.add(out)
        alias_param = f"{out}_alias"

        # Use .add() for atomic_add mode, .set() otherwise
        scatter_op = "add" if mode == "atomic_add" else "set"

        if is_point_scatter:
            # Single-element scatter
            indirect_var = scatter_info["indirect_var"]
            indirect_dim = scatter_info["indirect_dim"]
            output_shape = scatter_info["output_shape"]

            # Build index tuple with 0s for other dimensions
            index_parts = []
            for dim in range(len(output_shape)):
                if dim == indirect_dim:
                    index_parts.append(indirect_var)
                else:
                    index_parts.append("0")

            index_tuple = ", ".join(index_parts)
            return f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"

        # Scatter with iteration variables
        indirect_var = scatter_info["indirect_var"]
        dims_before = scatter_info["dims_before"]
        dims_after = scatter_info["dims_after"]

        # Determine if element-wise or slice-based scatter
        buf = V.graph.get_buffer(name)
        output_ndim = len(buf.get_size()) if buf is not None else 0

        num_iter_vars_in_store = len(dims_before) + len(dims_after)
        total_kernel_iter_vars = len(self.range_tree_nodes)
        remaining_dims = output_ndim - 1  # dims other than indirect

        is_element_wise = (
            num_iter_vars_in_store == remaining_dims
            and num_iter_vars_in_store == total_kernel_iter_vars
        )

        if is_element_wise:
            # Element-wise scatter: use iteration variable names
            index_parts = [var_name for var_name, size in dims_before]

            # Reshape indirect var for broadcasting if needed
            n_leading = len(dims_before)
            n_trailing = len(dims_after)
            if n_leading > 0 and n_trailing > 0:
                leading_ones = "None, " * n_leading
                trailing_nones = ", None" * n_trailing
                indirect_reshaped = f"{indirect_var}[{leading_ones}...{trailing_nones}]"
            else:
                indirect_reshaped = indirect_var
            index_parts.append(indirect_reshaped)

            index_parts.extend(var_name for var_name, size in dims_after)
        else:
            # Slice-based scatter: use : for iteration dimensions
            index_parts = [":" for _ in dims_before]
            # Flatten indirect variable to 1D for JAX scatter semantics.
            # The variable may have extra dims from _maybe_reshape_for_expand
            # (e.g., shape (601, 1) instead of (601,)), which would cause JAX
            # to expect a different value shape for broadcasting.
            index_parts.append(f"{indirect_var}.reshape(-1)")
            index_parts.extend(":" for _ in dims_after)

        index_tuple = ", ".join(index_parts)
        return (
            f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"
        )

    @typing_extensions.override
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        # Track the load index expression for argmax/argmin axis detection
        self.load_index_exprs[name] = index

        # Get base index expression
        index_str, needs_flatten = self._get_index_expr(index)

        # Check for buffer size mismatch requiring strided indexing
        index_str, needs_flatten = self._needs_strided_indexing(
            name, index, index_str, needs_flatten
        )

        # Adjust index for buffer shape (scalar, multi-dim, etc.)
        index_str, needs_flatten = self._adjust_index_for_buffer_shape(
            name, index, index_str, needs_flatten
        )

        # Build the load expression
        load_expr = self._build_load_expr(buf, name, index, index_str, needs_flatten)

        # Handle intermediate buffer squeezing for correct broadcasting
        if not needs_flatten and index_str == "...":
            load_expr = self._maybe_squeeze_intermediate_buffer(name, load_expr)
            # Reshape intermediate buffers to match canonical output shape
            # This handles cases like embedding + residual where buffer is (32, 64)
            # but should be (2, 16, 64) to match the kernel's output shape.
            load_expr = self._maybe_reshape_intermediate_buffer(name, load_expr)
            # Handle 1D buffer broadcasting for higher-dimensional kernels
            load_expr = self._maybe_broadcast_1d_buffer(name, index, load_expr)
            # Handle expand patterns (stride=0 dimensions) by inserting singleton dims
            load_expr = self._maybe_reshape_for_expand(name, index, load_expr)

        result = self.cse.generate(
            self.compute,
            load_expr,
            dtype=dtype,
        )

        # Track the shape of this variable for on-demand reshape at binary ops
        self._track_var_shape(str(result), name, load_expr, index)

        return result

    def _handle_mixed_indexing(self, index: sympy.Expr) -> str:
        """
        Handle indexing with both indirect variables and iteration variables.

        For example, x[indices, :] generates index = i0 + stride * tmp0
        where tmp0 is loaded from indices and i0 is the iteration variable.

        We need to convert this to JAX advanced indexing with proper broadcasting.
        When there are multiple iteration variables, they need different shapes
        to form an outer product (grid) rather than broadcasting together.

        Special case: For gather operations where a single iteration variable
        and single indirect variable have the same extent, they should be
        element-wise aligned, not broadcast into an outer product.

        PyTorch advanced indexing semantics: When multiple indirect indices have
        the same shape, they are paired element-wise (not outer product), and
        the combined result dimension appears at the FRONT of the output.
        """
        used_iter_vars_set = self._get_used_iter_vars(index)

        if len(used_iter_vars_set) == 0:
            return self.kexpr(index)

        # Sort iteration variables by their coefficient (stride) in the index expression.
        # Variables with larger strides correspond to earlier output dimensions.
        def get_coefficient(var):
            """Extract the coefficient of a variable in the index expression."""
            coeff = index.coeff(var)
            if coeff == 0:
                # Variable appears in a more complex form, try differentiation
                coeff = sympy.diff(index, var)
            # Convert to int if possible for sorting
            try:
                return int(coeff)
            except (TypeError, ValueError):
                # Symbolic coefficient - treat as outer dimension
                return float("inf")

        used_iter_vars = sorted(used_iter_vars_set, key=get_coefficient, reverse=True)
        iter_coeffs = [get_coefficient(var) for var in used_iter_vars]

        # Rename symbolic sizes to kernel parameter names
        index_str = self.kexpr(self.rename_indexing(index))
        indirect_var_syms = self._get_indirect_vars(index)
        indirect_vars = [str(sym) for sym in indirect_var_syms]

        # Get coefficients for indirect vars to determine output ordering
        indirect_coeffs = {str(s): get_coefficient(s) for s in indirect_var_syms}

        # Special case: reduction var + single indirect var = element-wise gather
        # Reduction vars (r prefix) iterate over the reduction dimension, and when paired
        # with an indirect var, both are aligned to that dimension (element-wise).
        # Pointwise vars form output dimensions and need the complex reshape code.
        if len(used_iter_vars) == 1 and len(indirect_vars) == 1:
            var = used_iter_vars[0]
            var_name = str(var)
            is_reduction_var = (
                var in self.range_tree_nodes and self.range_tree_nodes[var].is_reduction
            )

            if is_reduction_var:
                # Reduction var: simple element-wise gather
                if var in self.range_tree_nodes:
                    range_entry = self.range_tree_nodes[var]
                    range_size = range_entry.length
                    # Rename to use kernel parameter names for symbolic sizes
                    renamed_size = self.rename_indexing(range_size)
                    arange_expr = f"jnp.arange({self.kexpr(renamed_size)})"
                    index_str = index_str.replace(var_name, arange_expr)
                # Reshape indirect var for proper broadcasting with reduction var
                # e.g., tmp5 shape (2,16) + arange(64) shape (64,) won't broadcast
                # but tmp5[..., None] shape (2,16,1) + (64,) -> (2,16,64) works
                # However, if indirect var is already 1D, adding [..., None] would
                # create wrong broadcast: (N,1) + (N,) -> (N,N) instead of (N,)
                indirect_var = indirect_vars[0]
                indirect_shape = self.var_shapes.get(indirect_var)

                # Check if the kernel has any pointwise (non-reduction) iteration vars
                # If all vars are reduction vars (pure reduction kernel), the indirect
                # var is element-wise with the reduction, so no broadcast needed
                all_iter_vars = self._get_iter_vars()
                has_pointwise_vars = any(
                    not self.range_tree_nodes[v].is_reduction
                    for v in all_iter_vars
                    if v in self.range_tree_nodes
                )

                # Only add [..., None] if:
                # 1. We know the indirect var is multi-dimensional, OR
                # 2. Shape is unknown AND kernel has pointwise vars (need broadcast)
                if len(indirect_shape or ()) > 1 or (
                    indirect_shape is None and has_pointwise_vars
                ):
                    index_str = index_str.replace(
                        indirect_var, f"{indirect_var}[..., None]"
                    )
                # else: 1D indirect var or pure reduction kernel - element-wise
                return index_str
            # For pointwise vars, fall through to the complex reshape code

        # Check if multiple indirect vars should be paired element-wise.
        # In PyTorch, when multiple advanced indices have the same shape, they pair up.
        # The paired dimension goes to the FRONT of the output.
        # However, if indirect vars have different shapes (e.g., (1,4) and (4,1)),
        # they form an outer product instead.
        # We detect element-wise pairing when:
        # 1. Multiple indirect vars exist
        # 2. There's exactly ONE unused iteration variable (for the shared paired dim)
        # For outer product, there are MULTIPLE unused iter vars (one per indirect dim)
        paired_indirect = False
        if len(indirect_vars) > 1:
            # Count unused iteration variables (defined but not in index expression)
            unused_iter_vars = self._get_iter_vars() - used_iter_vars_set
            # Element-wise pairing: one unused iter var for the shared paired dimension
            # Outer product: multiple unused iter vars (one for each indirect var dimension)
            paired_indirect = len(unused_iter_vars) == 1

        if paired_indirect:
            # Multiple indirect vars with element-wise pairing
            # Output order: (paired_indirect_dim, iter_var_dims...)
            # All indirect vars get the same shape: (N, 1, 1, ...) for first dim
            # Iter vars come after: second dim onwards

            # Count total output dims: 1 (paired) + len(iter_vars) for non-newaxis
            # But some iter vars may be for newaxis dimensions (size 1)
            n_output_dims = 1 + len(used_iter_vars)

            # Reshape indirect vars to occupy the first dimension
            for indirect_var in indirect_vars:
                trailing_ones = ", 1" * len(used_iter_vars)
                reshape_expr = f"{indirect_var}.reshape(-1{trailing_ones})"
                index_str = index_str.replace(indirect_var, reshape_expr)

            # Reshape iteration variables to occupy subsequent dimensions
            # Sort by coefficient (descending) to determine order
            for i, var in enumerate(used_iter_vars):
                var_name = str(var)
                if var in self.range_tree_nodes:
                    range_entry = self.range_tree_nodes[var]
                    range_size = range_entry.length
                    # Rename to use kernel parameter names for symbolic sizes
                    renamed_size = self.rename_indexing(range_size)

                    # Shape: (1, ..., N, ..., 1) where N is at position i+1
                    # Position 0 is for paired indirect vars
                    shape_parts = ["1"] * n_output_dims
                    shape_parts[i + 1] = self.kexpr(renamed_size)
                    shape_str = ", ".join(shape_parts)
                    arange_expr = (
                        f"jnp.arange({self.kexpr(renamed_size)}).reshape({shape_str})"
                    )

                    index_str = index_str.replace(var_name, arange_expr)

            return index_str

        # Single indirect var case (or no indirect vars handled above)
        # Build a sorted list of all components by coefficient (descending)
        # Each component is (coeff, type, var) where type is 'iter' or 'indirect'
        all_components = []
        for var in used_iter_vars:
            all_components.append((get_coefficient(var), "iter", var))
        for sym in indirect_var_syms:
            all_components.append((get_coefficient(sym), "indirect", sym))
        all_components.sort(key=lambda x: x[0], reverse=True)

        # Calculate trailing dims needed for each component
        # Each component needs trailing dims for all subsequent iter vars
        # plus trailing dims for all dimensions of subsequent indirect vars
        # For simplicity, assume each indirect var contributes some dimensions
        # that will be handled by the reshape at store time

        # Check which indirect vars are already properly shaped (derived from iter vars)
        # These don't need additional reshaping and shouldn't affect iter var trailing dims
        indirect_var_ndims: dict[str, int] = {}
        already_shaped_indirect: set[str] = set()

        # Get total number of iteration dims in the kernel (not just those used in index)
        # Indirect vars computed within the kernel will have this many dimensions
        all_iter_vars = self._get_iter_vars()
        n_total_iter_dims = len(all_iter_vars)
        n_used_iter_dims = len(used_iter_vars)

        for indirect_var in indirect_vars:
            indirect_shape = self.var_shapes.get(indirect_var)
            indirect_coeff = indirect_coeffs.get(indirect_var, 0)

            # Check if any iter var has smaller coefficient (comes after this indirect var)
            # If so, the indirect var needs trailing dims and is NOT "already shaped"
            has_trailing_iter_vars = any(c < indirect_coeff for c in iter_coeffs)

            if indirect_shape is not None:
                ndim = len(indirect_shape)
                indirect_var_ndims[indirect_var] = ndim
                # Only mark as already shaped if:
                # 1. ndim matches total iter dims AND
                # 2. No iter vars need to come after (no trailing dims needed)
                if ndim == n_total_iter_dims and not has_trailing_iter_vars:
                    already_shaped_indirect.add(indirect_var)
            else:
                # Unknown shape - use n_used_iter_dims as default
                # Previously we assumed "already shaped" when n_total_iter_dims > n_used_iter_dims,
                # but this was too aggressive and caused inconsistent trailing dims
                # for different index expressions using similar variables (e.g., tmp9 with
                # known shape vs tmp15 with unknown shape derived from tmp9).
                # Using n_used_iter_dims provides consistency: vars appearing in index
                # expressions with the same iter vars get the same trailing dims.
                # We also don't mark as already_shaped - treat similar to known-shape vars.
                indirect_var_ndims[indirect_var] = max(1, n_used_iter_dims)

        # For iter vars, we need to count how many dimensions come after in the output
        for i, var in enumerate(used_iter_vars):
            var_name = str(var)
            if var in self.range_tree_nodes:
                range_entry = self.range_tree_nodes[var]
                range_size = range_entry.length
                # Rename to use kernel parameter names for symbolic sizes
                renamed_size = self.rename_indexing(range_size)
                var_coeff = get_coefficient(var)

                arange_expr = f"jnp.arange({self.kexpr(renamed_size)})"

                # Count trailing dims needed:
                # - One for each subsequent iter var (with smaller coeff)
                # - For already-shaped indirect vars: count their dimensions minus
                #   dimensions already covered by other iter vars
                n_trailing_iter = sum(1 for c in iter_coeffs if c < var_coeff)

                # For already-shaped indirect vars (derived from other iter vars),
                # we need trailing dims for the dimensions they occupy
                n_trailing_indirect = 0
                for ind_var, ind_coeff in indirect_coeffs.items():
                    if ind_coeff < var_coeff:
                        if ind_var in already_shaped_indirect:
                            # Already shaped from other iter vars - need trailing dims
                            # for the dimensions not used in this index expression
                            n_other_iter_dims = n_total_iter_dims - n_used_iter_dims
                            n_trailing_indirect += n_other_iter_dims
                        else:
                            # External indirect var - use its actual ndim
                            n_trailing_indirect += indirect_var_ndims.get(ind_var, 1)
                n_trailing = n_trailing_iter + n_trailing_indirect

                if n_trailing > 0:
                    trailing_dims = ", None" * n_trailing
                    arange_expr = f"{arange_expr}[:{trailing_dims}]"

                index_str = index_str.replace(var_name, arange_expr)

        # Reshape indirect variables for proper broadcasting.
        # Skip reshaping for indirect vars that are already properly shaped.
        for indirect_var in indirect_vars:
            if indirect_var in already_shaped_indirect:
                # Already has correct shape from being derived from iter vars
                continue

            indirect_coeff = indirect_coeffs[indirect_var]

            # Count dims needed before and after this indirect var
            n_leading = sum(1 for c in iter_coeffs if c > indirect_coeff)
            n_trailing = sum(1 for c in iter_coeffs if c < indirect_coeff)

            # Build the indexing expression with leading Nones, ellipsis, trailing Nones
            if n_leading > 0 and n_trailing > 0:
                leading_nones = "None, " * n_leading
                trailing_nones = ", None" * n_trailing
                reshape_expr = f"{indirect_var}[{leading_nones}...{trailing_nones}]"
            elif n_leading > 0:
                leading_nones = "None, " * n_leading
                reshape_expr = f"{indirect_var}[{leading_nones}...]"
            elif n_trailing > 0:
                trailing_nones = ", None" * n_trailing
                reshape_expr = f"{indirect_var}[...{trailing_nones}]"
            else:
                reshape_expr = indirect_var

            index_str = index_str.replace(indirect_var, reshape_expr)

        return index_str

    @typing_extensions.override
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:
        # mode can be None (set), "atomic_add" (accumulate), etc.
        if mode is not None and mode != "atomic_add":
            raise Unsupported(f"pallas store mode '{mode}' not supported")
        out = self.args.output(name)
        self.store_buffer_names.add(name)

        # Check if this is a scalar output (reduction to scalar)
        buf = V.graph.get_buffer(name)
        is_scalar = buf is not None and len(buf.get_size()) == 0

        if is_scalar:
            # For scalar outputs, use jnp.full to handle shape mismatch
            store_expr = (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.asarray({value}).reshape({out}.shape))"
            )
        else:
            # Check for scatter pattern (indirect indexing for stores)
            scatter_info = self._detect_scatter_pattern(index, name)

            if scatter_info is not None:
                store_expr = self._build_scatter_store_expr(
                    out, value, scatter_info, name, mode
                )
            else:
                # Get base index expression
                index_str, needs_flatten = self._get_index_expr(index)

                # Check for im2col-like patterns
                index_str, needs_flatten = self._check_im2col_pattern(
                    index, index_str, needs_flatten
                )

                # Build the store expression
                store_expr = self._build_store_expr(
                    out, name, index, value, index_str, needs_flatten, mode
                )

        self.stores.writeline(store_expr)
        # Track which output param this store uses for filtering in codegen_kernel
        self.store_with_output.append((out, store_expr))

    def _get_index_coefficient(self, index: sympy.Expr, var: sympy.Symbol) -> int:
        """Get integer coefficient of a variable in an index expression."""
        coeff = index.coeff(var)
        if coeff == 0:
            coeff = sympy.diff(index, var)
        try:
            return int(coeff)
        except (TypeError, ValueError):
            return 0

    def _detect_scatter_pattern(
        self, index: sympy.Expr, output_name: str = ""
    ) -> Optional[dict[str, Any]]:
        """Detect scatter operation pattern. Returns scatter info dict or None."""
        indirect_syms = self._get_indirect_vars(index)
        if len(indirect_syms) != 1:
            return None

        indirect_sym = indirect_syms[0]
        indirect_var = str(indirect_sym)
        indirect_coeff = self._get_index_coefficient(index, indirect_sym)
        if indirect_coeff == 0:
            return None

        # Point scatter: no iteration variables, just indirect indexing
        if not self._has_iteration_vars(index):
            return self._detect_point_scatter(output_name, indirect_var, indirect_coeff)

        # Regular scatter: has both indirect and iteration variables
        return self._detect_iter_scatter(index, indirect_var, indirect_coeff)

    def _detect_point_scatter(
        self, output_name: str, indirect_var: str, indirect_coeff: int
    ) -> Optional[dict[str, Any]]:
        """Detect single-element scatter pattern."""
        if not output_name:
            return None
        try:
            buf = V.graph.get_buffer(output_name)
            output_shape = [int(s) for s in buf.get_size()]
        except Exception:
            return None

        if len(output_shape) < 2:
            return None

        # Find which dimension indirect var indexes based on coefficient
        cumulative = 1
        indirect_dim = len(output_shape) - 1
        for dim in range(len(output_shape) - 1, -1, -1):
            if indirect_coeff == cumulative:
                indirect_dim = dim
                break
            cumulative *= output_shape[dim]

        return {
            "indirect_var": indirect_var,
            "indirect_dim": indirect_dim,
            "dims_before": [],
            "dims_after": [],
            "is_point_scatter": True,
            "output_shape": output_shape,
        }

    def _detect_iter_scatter(
        self, index: sympy.Expr, indirect_var: str, indirect_coeff: int
    ) -> Optional[dict[str, Any]]:
        """Detect scatter pattern with iteration variables."""
        used_iter_vars = self._get_used_iter_vars(index)

        # Collect (var_name, coefficient, length) for each variable
        all_vars = []
        for var in used_iter_vars:
            coeff = self._get_index_coefficient(index, var)
            if coeff > 0 and var in self.range_tree_nodes:
                length = self._safe_int(self.range_tree_nodes[var].length)
                if length is None:
                    return None
                all_vars.append((str(var), coeff, length))

        all_vars.append((indirect_var, indirect_coeff, -1))
        all_vars.sort(key=lambda x: x[1], reverse=True)

        # Find indirect variable position
        indirect_pos = next(
            (i for i, (name, _, _) in enumerate(all_vars) if name == indirect_var),
            None,
        )
        if indirect_pos is None:
            return None

        # Verify coefficients form valid stride pattern
        expected = 1
        for _, coeff, length in reversed(all_vars[indirect_pos + 1 :]):
            if coeff != expected:
                return None
            expected *= length
        if indirect_coeff != expected:
            return None

        return {
            "indirect_var": indirect_var,
            "indirect_dim": indirect_pos,
            "dims_before": [(n, l) for n, _, l in all_vars[:indirect_pos]],
            "dims_after": [(n, l) for n, _, l in all_vars[indirect_pos + 1 :]],
            "is_point_scatter": False,
            "output_shape": None,
        }

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:  # type: ignore[override]
        """
        Generate code for reduction operations in JAX/Pallas.

        Reductions in Pallas work by:
        1. Loading the input data into the kernel
        2. Applying JAX reduction operations (jnp.sum, jnp.max, etc.)
        3. Storing the reduced result

        The reduction happens over the loaded block of data.
        """
        assert self.inside_reduction

        # Handle welford_reduce using the fallback (computes via sum reductions)
        if reduction_type == "welford_reduce":
            return self.welford_reduce_fallback(dtype, value)

        if isinstance(value, tuple):
            raise Unsupported(
                "Tuple reductions (e.g., welford_combine) not supported in Pallas backend"
            )

        # Check if this reduction is already cached
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        # Map reduction types to JAX functions
        reduction_ops = {
            "sum": "jnp.sum",
            "prod": "jnp.prod",  # CPU only - not supported in Pallas GPU (Mosaic) backend
            "max": "jnp.max",
            "min": "jnp.min",
            "any": "jnp.any",
            "argmax": "jnp.argmax",
            "argmin": "jnp.argmin",
        }

        # Determine if this is a partial reduction (has pointwise dimensions)
        # or a full reduction to scalar
        pointwise_prefixes = OrderedSet(["x", "y", "z"])
        has_pointwise = any(p in self.numels for p in pointwise_prefixes)

        # Get the pointwise and reduction numels
        pointwise_numel: Optional[int] = self._compute_prefix_numel(pointwise_prefixes)
        reduction_numel: Optional[int] = self._compute_reduction_numel()

        # Count the number of pointwise and reduction dimensions
        n_reduction_dims = sum(
            1 for var, entry in self.range_tree_nodes.items() if entry.is_reduction
        )

        if reduction_type == "xor_sum":
            if has_pointwise and pointwise_numel and reduction_numel:
                reduction_expr = f"jnp.bitwise_xor.reduce({value}.reshape({pointwise_numel}, -1), axis=-1)"
            else:
                reduction_expr = f"jnp.bitwise_xor.reduce({value})"
        elif reduction_type in ("argmax", "argmin"):
            # For argmax/argmin, the result is indices into the reduction dimension.
            # Unlike sum/max/min, we can't just reshape because the indices depend
            # on which axis we reduce over. We need to determine the correct axis.
            reduction_op = reduction_ops[reduction_type]
            # Check if this is a true partial reduction (pointwise numel > 1)
            # When pointwise_numel == 1, it's effectively a full reduction to scalar
            is_partial_reduction = (
                has_pointwise
                and pointwise_numel
                and pointwise_numel > 1
                and reduction_numel
            )
            # argmax/argmin doesn't use symbolic partial reduction logic
            is_symbolic_partial = False
            if is_partial_reduction and n_reduction_dims > 0:
                # Partial reduction: determine the reduction axis from load index
                # The reduction variable's coefficient in the index expression tells us its stride
                # Higher stride = outer axis (lower axis number in row-major order)
                reduction_axis = -1  # Default to last axis
                if self.load_index_exprs:
                    # Get the first load index expression
                    load_index = next(iter(self.load_index_exprs.values()))
                    # Find the reduction variable (starts with 'r')
                    reduction_vars = [
                        var
                        for var, entry in self.range_tree_nodes.items()
                        if entry.is_reduction
                    ]
                    if reduction_vars:
                        r_var = reduction_vars[0]
                        # Get the coefficient (stride) of the reduction variable
                        r_coeff = load_index.coeff(r_var)
                        r_stride = self._safe_int(r_coeff) if r_coeff != 0 else 1
                        if r_stride is None:
                            r_stride = 1
                        # Get pointwise variable
                        pw_vars = [
                            var
                            for var, entry in self.range_tree_nodes.items()
                            if not entry.is_reduction
                        ]
                        if pw_vars:
                            pw_var = pw_vars[0]
                            pw_coeff = load_index.coeff(pw_var)
                            pw_stride = self._safe_int(pw_coeff) if pw_coeff != 0 else 1
                            if pw_stride is None:
                                pw_stride = 1
                            # Higher stride = earlier (outer) axis
                            # For 2D: axis 0 has stride = dim1_size, axis 1 has stride = 1
                            reduction_axis = 0 if r_stride > pw_stride else -1
                reduction_expr = f"{reduction_op}({value}, axis={reduction_axis})"
            else:
                # Full reduction to scalar
                reduction_expr = f"{reduction_op}({value})"
        elif reduction_type in reduction_ops:
            # Check for true partial reduction (pointwise_numel > 1 means we have
            # actual pointwise dimensions, not just a scalar placeholder)
            is_partial_reduction = (
                has_pointwise
                and pointwise_numel is not None
                and pointwise_numel > 1
                and reduction_numel
            )
            # Also check for symbolic partial reduction (has both pw and reduction vars)
            is_symbolic_partial = (
                has_pointwise and n_reduction_dims > 0 and pointwise_numel is None
            )
            if is_partial_reduction:
                # For partial reductions, we need to:
                # 1. Find which axes are reduction axes (contiguous axes whose product = reduction_numel)
                # 2. Move pointwise axes to front, reduction axes to back
                # 3. Reshape to (pointwise_numel, reduction_numel) and reduce over last axis
                # 4. Reshape output with 1s in reduced dims for proper broadcasting
                reduction_op = reduction_ops[reduction_type]
                # Use a helper to find reduction axes by product matching
                reduction_expr = f"_pallas_partial_reduce({reduction_op}, {value}, {pointwise_numel}, {reduction_numel})"
            elif is_symbolic_partial:
                # Symbolic sizes: use axis-based reduction (axis=0 for outer reduction)
                reduction_expr = f"{reduction_ops[reduction_type]}({value}, axis=0)"
            else:
                # Full reduction to scalar
                reduction_expr = f"{reduction_ops[reduction_type]}({value})"
        else:
            raise Unsupported(
                f"Reduction type '{reduction_type}' not yet supported in Pallas backend. "
                f"Supported types: {list(reduction_ops.keys())}, xor_sum"
            )

        # Generate CSE variable for the reduction result
        result = self.cse.generate(
            self.compute,
            reduction_expr,
            dtype=dtype,
        )

        # Track reduction output shape for binary op broadcasting
        # For partial reductions with keepdims, output shape is (pointwise_numel, 1)
        # flattened, which will be reshaped to canonical output with 1s in reduced dims
        if is_partial_reduction:
            # Output shape is (pointwise_numel, 1) - pointwise dims with 1 for reduced dim
            self.var_shapes[reduction_expr] = (pointwise_numel, 1)
            self.var_shapes[str(result)] = (pointwise_numel, 1)
        elif is_symbolic_partial:
            # Symbolic: we don't know exact shape, but it's a partial reduction
            pass  # Can't track precisely
        else:
            # Full reduction to scalar
            self.var_shapes[reduction_expr] = ()
            self.var_shapes[str(result)] = ()

        # Cache the result
        self.cse.reduction_cache[cache_key] = result
        return result

    @staticmethod
    def _buffer_is_contiguous(buffer_name: str) -> bool:
        buf = V.graph.get_buffer(buffer_name)
        layout = buf.get_layout()
        return layout.is_contiguous()

    def codegen_kernel(self, name: Optional[str] = None) -> str:  # type: ignore[override]
        """
        Generate the complete Pallas kernel code as a Python string.

        This includes:
        - Import statements for JAX/Pallas
        - The kernel function that operates on refs
        - The main wrapper function that handles PyTorch<->JAX conversions via DLPack

        Args:
            name: Optional kernel name (will use placeholder if not provided)

        Returns:
            str: Complete Python source code for the Pallas kernel
        """
        code = IndentedBuffer()

        # Define the Pallas kernel: accepts refs, uses broadcasted expressions
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_params if p.startswith("out_ptr")]
        output_params = [
            p for p in kernel_params if p.startswith(("out_ptr", "in_out_ptr"))
        ]
        # Identify size variable parameters (scalars like load_seed_offset)
        size_var_names = OrderedSet(self.args.sizevars.values())
        size_var_params = [p for p in kernel_params if p in size_var_names]
        if not output_params:
            raise RuntimeError("Pallas backend requires at least one output buffer")

        output_buffer_lookup = {
            inner: outer
            for outer, inner in self.args.output_buffers.items()
            if isinstance(inner, str)
        }

        kernel_name = name or "<KERNEL_NAME>"
        interpret_is_cpu = V.graph.get_current_device_or_throw().type == "cpu"
        is_tpu = torch._inductor.config._debug_cpu_to_tpu_pallas
        if is_tpu:
            if not torch._inductor.config.pallas_take_first_jax_device_only:
                raise RuntimeError(
                    "Pallas backend currently only supports using the first JAX device."
                )
            if not has_tpu_pallas():
                raise RuntimeError(
                    "PALLAS_TARGET_TPU is set, but no TPU device was found. "
                    "Please make sure that you have a TPU available and that JAX is configured correctly."
                )
        interpret_literal = "True" if interpret_is_cpu else "False"

        # For GPU (Mosaic backend), import plgpu for TMA operations
        # Import math for symbolic expressions (e.g., math.floor, math.log2)
        imports = """
import functools
import math
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime
def _pallas_partial_reduce(reduce_fn, v, pw_numel, red_numel):
    # Helper for partial reductions: reorders axes and reduces
    # Returns result with keepdims-style shape for proper in-kernel broadcasting
    shape = tuple(v.shape)
    # Find contiguous axes whose product = red_numel (search from right)
    red_axes = None
    for i in range(len(shape) - 1, -1, -1):
        prod = 1
        for j in range(i, -1, -1):
            prod *= shape[j]
            if prod == red_numel:
                red_axes = list(range(j, i + 1))
                break
        if red_axes is not None:
            break
    if red_axes is None:
        red_axes = [len(shape) - 1]
    # Build output shape with 1s for reduced dimensions (keepdims style)
    out_shape = tuple(1 if i in red_axes else s for i, s in enumerate(shape))
    # Move pointwise axes to front, reduction axes to back
    pw_axes = [i for i in range(len(shape)) if i not in red_axes]
    reordered = jnp.moveaxis(v, pw_axes, list(range(len(pw_axes))))
    result = reduce_fn(reordered.reshape(pw_numel, red_numel), axis=-1)
    return result.reshape(out_shape)
def _pallas_expand_for_broadcast(v, target_shape):
    # Helper for expand patterns: reshape value to be broadcastable to target_shape.
    # For expand (stride=0), value has fewer elements. We insert singleton dims
    # as needed for broadcast compatibility.
    # E.g., (2, 16, 2, 16) -> (2, 16, 2, 1, 16) to broadcast to (2, 16, 2, 2, 16)
    v_shape = list(v.shape)
    t_shape = list(target_shape)
    v_numel = 1
    for s in v_shape:
        v_numel *= s
    t_numel = 1
    for s in t_shape:
        t_numel *= s
    # Case 1: Same numel - just reshape directly
    if v_numel == t_numel:
        return v.reshape(target_shape)
    # Case 2: Same ndim but different numel (view + expand pattern)
    # E.g., input (2, 16, 2, 8, 2) with 1024 elements -> target (2, 16, 2, 2, 16) with 2048
    # The input needs to be reshaped to match the target with 1s at expanded positions.
    # Key insight: find common prefix between input and target shapes, then work on suffix.
    if len(v_shape) == len(t_shape) and v_numel < t_numel:
        # Find how many leading dimensions match
        common_prefix_len = 0
        for i in range(len(t_shape)):
            if i < len(v_shape) and v_shape[i] == t_shape[i]:
                common_prefix_len = i + 1
            else:
                break
        # Product of remaining v dims after the common prefix
        v_remaining = 1
        for s in v_shape[common_prefix_len:]:
            v_remaining *= s
        # Build intermediate shape: prefix + suffix with 1s at expansion positions
        prefix = list(t_shape[:common_prefix_len])
        suffix = list(t_shape[common_prefix_len:])
        # Find which suffix dims to set to 1 to match v_remaining
        for i in range(len(suffix)):
            test_suffix = list(suffix)
            test_suffix[i] = 1
            test_numel = 1
            for s in test_suffix:
                test_numel *= s
            if test_numel == v_remaining:
                intermediate = prefix + test_suffix
                return jnp.broadcast_to(v.reshape(-1).reshape(intermediate), target_shape)
        # Try combinations of 2 dims in suffix
        for i in range(len(suffix)):
            for j in range(i + 1, len(suffix)):
                test_suffix = list(suffix)
                test_suffix[i] = 1
                test_suffix[j] = 1
                test_numel = 1
                for s in test_suffix:
                    test_numel *= s
                if test_numel == v_remaining:
                    intermediate = prefix + test_suffix
                    return jnp.broadcast_to(v.reshape(-1).reshape(intermediate), target_shape)
    # Case: Right-aligned expand (PyTorch expand semantics)
    # E.g., (2, 1, 2) -> (2, 1, 2, 3, 2): value dims align to target's TRAILING dims
    # Intermediate: [1, 1, 2, 1, 2] (prepend 1s, keep value dims)
    if len(v_shape) < len(t_shape) and v_numel < t_numel:
        extra_dims = len(t_shape) - len(v_shape)
        trailing_target = list(t_shape[extra_dims:])
        # Check if value dims are broadcast-compatible with trailing target dims
        valid = True
        for v_dim, t_dim in zip(v_shape, trailing_target):
            if v_dim != t_dim and v_dim != 1:
                valid = False
                break
        if valid:
            intermediate = [1] * extra_dims + list(v_shape)
            return jnp.broadcast_to(v.reshape(intermediate), target_shape)
    # Case 3: Expand case - use value's shape to determine broadcast pattern
    # The value's shape encodes the expand info: dims with size 1 are broadcast dims.
    # E.g., value (64, 1, 16) -> target (2, 16, 2, 2, 16):
    #   - Value dim 1 has size 1 -> this is the expand dimension
    #   - Value dim 0 (64) = target dims 0,1,2 merged (2*16*2)
    #   - Value dim 2 (16) = target dim 4
    #   - Intermediate shape: (2, 16, 2, 1, 16)
    if v_numel < t_numel:
        # Find singleton (size=1) dimensions in value - these are expand dims
        v_singletons = [i for i, s in enumerate(v_shape) if s == 1]
        if v_singletons:
            # Build intermediate shape by matching value dims to target dims
            # Strategy: greedily match from both ends, insert 1s for singletons
            intermediate = []
            v_idx = 0
            t_idx = 0
            while t_idx < len(t_shape):
                if v_idx >= len(v_shape):
                    # No more value dims - remaining target dims should be 1s
                    intermediate.append(1)
                    t_idx += 1
                elif v_idx in v_singletons:
                    # This value dim is a singleton (expand dim)
                    intermediate.append(1)
                    v_idx += 1
                    t_idx += 1
                else:
                    # Try to match value dim to consecutive target dims
                    v_dim = v_shape[v_idx]
                    t_dim = t_shape[t_idx]
                    if v_dim == t_dim:
                        # Direct match
                        intermediate.append(t_dim)
                        v_idx += 1
                        t_idx += 1
                    elif v_dim > t_dim:
                        # Value dim is product of multiple target dims
                        prod = t_dim
                        intermediate.append(t_dim)
                        t_idx += 1
                        while t_idx < len(t_shape) and prod < v_dim:
                            # Check if next target dim is a singleton (expand)
                            if t_shape[t_idx] * prod == v_dim:
                                intermediate.append(t_shape[t_idx])
                                prod *= t_shape[t_idx]
                                t_idx += 1
                            elif v_idx + 1 < len(v_shape) and v_idx + 1 in v_singletons:
                                # Next value dim is singleton, insert 1 here
                                intermediate.append(1)
                                t_idx += 1
                                break
                            else:
                                intermediate.append(t_shape[t_idx])
                                prod *= t_shape[t_idx]
                                t_idx += 1
                        v_idx += 1
                    else:
                        # Can't match - insert 1 (assume expand dim)
                        intermediate.append(1)
                        t_idx += 1
            # Verify intermediate shape has correct numel
            inter_numel = 1
            for s in intermediate:
                inter_numel *= s
            if inter_numel == v_numel and len(intermediate) == len(t_shape):
                return jnp.broadcast_to(v.reshape(-1).reshape(intermediate), target_shape)
        # Case: Standard broadcasting with fewer dimensions
        # E.g., (256, 256) -> (256, 256, 256)
        # Standard NumPy/PyTorch semantics: prepend 1s on the left
        # ONLY applies when value's dims EXACTLY match target's TRAILING dims
        if len(v_shape) < len(t_shape):
            trailing_target = list(t_shape[-len(v_shape):])
            if v_shape == trailing_target:
                leading_ones = len(t_shape) - len(v_shape)
                intermediate = [1] * leading_ones + v_shape
                return jnp.broadcast_to(v.reshape(intermediate), target_shape)
        # Fallback: try each target dim as the expand dim (try LEFT to RIGHT first
        # to prefer standard broadcast semantics of prepending 1s)
        for i in range(len(t_shape)):
            test_shape = list(t_shape)
            test_shape[i] = 1
            test_numel = 1
            for s in test_shape:
                test_numel *= s
            if test_numel == v_numel:
                return jnp.broadcast_to(v.reshape(-1).reshape(test_shape), target_shape)
    # Case 4: Fewer dims in v (non-expand case) - insert singletons by matching dims
    if len(v_shape) < len(t_shape) and v_numel == t_numel:
        # Scan from left to right, matching dimensions.
        result_shape = []
        v_idx = 0
        for t_idx in range(len(t_shape)):
            if v_idx < len(v_shape) and v_shape[v_idx] == t_shape[t_idx]:
                result_shape.append(v_shape[v_idx])
                v_idx += 1
            else:
                result_shape.append(1)
        return jnp.broadcast_to(v.reshape(result_shape), target_shape)
    # Fall back to direct broadcast
    return jnp.broadcast_to(v, target_shape)
""" + (
            "\nfrom jax.experimental.pallas import mosaic_gpu as plgpu"
            if not interpret_is_cpu
            else ""
        )
        code.splice(imports, strip=True)

        aliasable_flags: dict[str, bool] = {}
        for param in pure_out_params:
            buffer_name = output_buffer_lookup.get(param)
            is_contiguous = buffer_name is not None and self._buffer_is_contiguous(
                buffer_name
            )
            # Enable aliasing if:
            # 1. Not on CPU and buffer is contiguous (normal case), OR
            # 2. Output needs to be readable (for scatter operations)
            # outputs_need_read contains output parameter names (e.g., out_ptr0)
            needs_read = param in self.outputs_need_read
            aliasable_flags[param] = (
                (not interpret_is_cpu) and is_contiguous
            ) or needs_read
        alias_params = [
            f"{param}_alias" for param in pure_out_params if aliasable_flags[param]
        ]
        pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        kernel_input_params = alias_params + pointer_tail
        full_kernel_params = alias_params + kernel_params
        non_alias_out_set = OrderedSet(
            [name for name, flag in aliasable_flags.items() if not flag]
        )
        # On CPU (interpret=True), we need to copy back even aliased outputs
        # because pallas_call returns a new array (doesn't mutate in-place)
        # For outputs that need read access (scatter), we enable aliasing to read
        # current values, but still need to copy back the result
        if interpret_is_cpu:
            # Copy back all outputs on CPU
            copy_output_indices = list(range(len(output_params)))
        else:
            copy_output_indices = [
                idx
                for idx, name in enumerate(output_params)
                if name in non_alias_out_set
            ]
        self.aliasable_out_ptrs = aliasable_flags

        # Generate kernel body into a separate buffer first.
        # This allows us to discover all size variables (registered via rename_indexing)
        # before generating the kernel signature.
        kernel_body = IndentedBuffer()
        with kernel_body.indent():
            # Generate iteration variables as jnp.arange arrays
            # These are used by index_expr operations like torch.arange
            # Skip on GPU - jnp.arange is not supported by Pallas Mosaic backend
            if self.range_tree_nodes and not self.is_gpu:
                kernel_body.writeline("# Define iteration variables as JAX arrays")

                # Find reshape target: N-D shape whose numel matches an iteration
                # var. Try output first (repeat/upsample), then inputs (reductions).
                iter_lengths = OrderedSet(
                    [
                        int(e.length)
                        for e in self.range_tree_nodes.values()
                        if isinstance(e.length, (int, sympy.Integer))
                    ]
                )

                def _get_nd_shape_if_matches(buf_name):
                    buf = V.graph.try_get_buffer(buf_name)
                    if buf is None or len(buf.get_size()) <= 1:
                        return None, None
                    shape = tuple(
                        int(s) if isinstance(s, (int, sympy.Integer)) else s
                        for s in buf.get_size()
                    )
                    numel = math.prod(shape)
                    return (shape, numel) if numel in iter_lengths else (None, None)

                # Candidate buffers: output first, then inputs
                candidate_buf_names = []
                if output_params:
                    buf_name = output_buffer_lookup.get(output_params[0])
                    if buf_name:
                        candidate_buf_names.append(buf_name)
                candidate_buf_names.extend(self.args.input_buffers)

                # Find first N-D buffer whose numel matches an iteration var
                reshape_target_shape, reshape_target_numel = None, None
                for name in candidate_buf_names:
                    result = _get_nd_shape_if_matches(name)
                    if result[0]:
                        reshape_target_shape, reshape_target_numel = result
                        break

                # Collect all iteration variable info for broadcasting shape computation
                var_items = list(self.range_tree_nodes.items())

                # Count vars that are NOT the "total" var (which equals output numel)
                # These are the actual iteration dimensions that need broadcasting
                broadcast_vars = []
                total_var_idx = None
                for idx, (var_sym, entry) in enumerate(var_items):
                    length_val = self._safe_int(entry.length)
                    if length_val is not None and length_val == reshape_target_numel:
                        total_var_idx = idx
                    else:
                        broadcast_vars.append((idx, var_sym, entry, length_val))

                num_broadcast_dims = len(broadcast_vars)

                for idx, (var_sym, entry) in enumerate(var_items):
                    var_name = str(var_sym)
                    length = entry.length
                    # Rename symbolic lengths to use kernel parameter names
                    renamed_length = self.rename_indexing(length)
                    length_str = self.kexpr(renamed_length)
                    length_val = self._safe_int(length)

                    # For symbolic lengths, only reshape if we have a valid target shape
                    # Without a target, we can't determine correct dimensions
                    if length_val is None:
                        if (
                            reshape_target_shape
                            and num_broadcast_dims > 1
                            and idx != total_var_idx
                        ):
                            # Symbolic var in multi-broadcast case needs reshape
                            broadcast_idx = next(
                                (
                                    i
                                    for i, (vidx, _, _, _) in enumerate(broadcast_vars)
                                    if vidx == idx
                                ),
                                None,
                            )
                            if broadcast_idx is not None:
                                # Same logic as concrete case
                                has_reduction_vars = any(
                                    str(v).startswith("r")
                                    for _, v, _, _ in broadcast_vars
                                )
                                has_pointwise_vars = any(
                                    not str(v).startswith("r")
                                    for _, v, _, _ in broadcast_vars
                                )
                                is_mixed = has_reduction_vars and has_pointwise_vars
                                if is_mixed:
                                    axis_idx = broadcast_idx
                                else:
                                    axis_idx = num_broadcast_dims - 1 - broadcast_idx
                                shape_parts = ["1"] * num_broadcast_dims
                                shape_parts[axis_idx] = length_str
                                shape_str = ", ".join(shape_parts)
                                arange = f"jnp.arange({length_str})"
                                kernel_body.writeline(
                                    f"{var_name} = {arange}.reshape({shape_str})"
                                )
                                continue
                        kernel_body.writeline(f"{var_name} = jnp.arange({length_str})")
                        continue

                    if (
                        reshape_target_shape
                        and len(reshape_target_shape) > 1
                        and length_val == reshape_target_numel
                    ):
                        # Reshape to match output/input shape for broadcasting
                        shape_str = ", ".join(str(s) for s in reshape_target_shape)
                        arange = f"jnp.arange({length_str})"
                        kernel_body.writeline(
                            f"{var_name} = {arange}.reshape({shape_str})"
                        )
                    elif num_broadcast_dims > 1 and idx != total_var_idx:
                        # Find position of this var among broadcast vars
                        broadcast_idx = next(
                            i
                            for i, (vidx, _, _, _) in enumerate(broadcast_vars)
                            if vidx == idx
                        )
                        # Reshape for broadcasting with other iteration vars.
                        # Axis placement depends on var types (reduction r* vs x*):
                        # - Mixed: pointwise first, reduction last for output reshape
                        # - Same-type: reverse order, first var innermost
                        has_reduction_vars = any(
                            str(v).startswith("r") for _, v, _, _ in broadcast_vars
                        )
                        has_pointwise_vars = any(
                            not str(v).startswith("r") for _, v, _, _ in broadcast_vars
                        )
                        is_mixed = has_reduction_vars and has_pointwise_vars
                        if is_mixed:
                            # Mixed kernel: pointwise vars first, reduction vars last
                            axis_idx = broadcast_idx
                        else:
                            # Same-type: reverse order (first var -> innermost)
                            axis_idx = num_broadcast_dims - 1 - broadcast_idx
                        shape_parts = ["1"] * num_broadcast_dims
                        shape_parts[axis_idx] = length_str
                        shape_str = ", ".join(shape_parts)
                        arange = f"jnp.arange({length_str})"
                        kernel_body.writeline(
                            f"{var_name} = {arange}.reshape({shape_str})"
                        )
                    else:
                        kernel_body.writeline(f"{var_name} = jnp.arange({length_str})")

            # Emit compute (CSE) and store lines; they reference *_ptr[index] directly.
            for line in self.compute._lines:
                kernel_body.writeline(str(line))

        # Recompute kernel parameters after kernel body generation.
        # Size variables may have been registered during kernel body generation
        # (e.g., via rename_indexing for symbolic sizes), so we need to re-fetch
        # the arg defs to capture all parameters including newly-registered size vars.
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        size_var_names = OrderedSet(self.args.sizevars.values())
        size_var_params = [p for p in kernel_params if p in size_var_names]
        pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        kernel_input_params = alias_params + pointer_tail
        full_kernel_params = alias_params + kernel_params

        # Now emit the kernel function with the correct signature
        kernel_signature = f"def {kernel_name}_kernel({', '.join(full_kernel_params)}):"
        code.writeline(kernel_signature)

        with code.indent():
            for line in kernel_body._lines:
                if isinstance(line, str):
                    # Remove any existing indentation and re-add with code's indentation
                    code.writeline(line.lstrip())
                else:
                    code._lines.append(line)

            # Add store lines (using recomputed full_kernel_params)
            # Filter stores to only emit those for outputs that are in kernel params.
            # This handles cases where an intermediate value was stored but the buffer
            # was later optimized away (not passed to the kernel).
            for out_ptr, store_line in self.store_with_output:
                if out_ptr in full_kernel_params:
                    code.writeline(store_line)

        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        donate_indices = []
        # Offset by 2 for (out_shapes, out_dtypes), plus size_var_params count
        base_offset = 2 + len(size_var_params)
        for idx, name in enumerate(kernel_input_params):
            if (name in alias_params) or name.startswith("in_out_ptr"):
                donate_indices.append(idx + base_offset)
        if donate_indices:
            donate_literal = "(" + ", ".join(str(x) for x in donate_indices) + ",)"
        else:
            donate_literal = "()"
        # Size variables are static args (after out_shapes and out_dtypes)
        static_argnums = list(range(2 + len(size_var_params)))
        static_argnums_literal = "(" + ", ".join(str(x) for x in static_argnums) + ",)"
        # For CPU interpret mode, add backend='cpu' to force CPU execution
        if interpret_is_cpu and not is_tpu:
            code.writeline(
                "@functools.partial("
                f"jax.jit, static_argnums={static_argnums_literal}, donate_argnums="
                f"{donate_literal}, backend='cpu')"
            )
        else:
            code.writeline(
                "@functools.partial("
                f"jax.jit, static_argnums={static_argnums_literal}, donate_argnums="
                f"{donate_literal})"
            )
        # Include size_var_params in wrapper signature
        wrapper_params = (
            ["out_shapes", "out_dtypes"] + size_var_params + kernel_input_params
        )
        code.writeline(f"def {jit_wrapper_name}({', '.join(wrapper_params)}):")
        with code.indent():
            code.writeline("out_specs = tuple(")
            code.writeline("    jax.ShapeDtypeStruct(shape, dtype)")
            code.writeline("    for shape, dtype in zip(out_shapes, out_dtypes)")
            code.writeline(")")

            alias_pairs: list[tuple[int, int]] = []
            for out_idx, name in enumerate(output_params):
                if name.startswith("out_ptr"):
                    if aliasable_flags.get(name, False):
                        alias_name = f"{name}_alias"
                        input_idx = kernel_input_params.index(alias_name)
                        alias_pairs.append((input_idx, out_idx))
                else:
                    input_idx = kernel_input_params.index(name)
                    alias_pairs.append((input_idx, out_idx))
            alias_map_literal = ", ".join(f"{i}: {o}" for (i, o) in alias_pairs)

            # Wrap kernel with functools.partial to pass scalar arguments (size variables)
            partial_args = []
            for sv_param in size_var_params:
                partial_args.append(f"{sv_param}={sv_param}")

            if partial_args:
                kernel_arg = f"functools.partial({kernel_name}_kernel, {', '.join(partial_args)}),"
            else:
                kernel_arg = f"{kernel_name}_kernel,"

            # Use plgpu.kernel for GPU (Mosaic), pl.pallas_call for CPU/TPU
            # TMA approach requires: no reductions, all inputs contiguous, same sizes
            use_tma = (
                self.is_gpu and self.use_emit_pipeline and self._can_use_tma_approach()
            )
            if use_tma:
                # Use lax.fori_loop with direct TMA for automatic OOB masking
                # TMA (Tensor Memory Accelerator) automatically handles out-of-bounds
                # accesses, eliminating the need for explicit padding to multiples of 128
                code.writeline("# Use lax.fori_loop with TMA for automatic OOB masking")
                code.writeline("from jax import lax")
                code.writeline("_tile_size = 128  # Warpgroup size")
                code.writeline("_orig_out_shapes = out_shapes")

                # Calculate max numel across all inputs/outputs for grid calculation
                code.writeline("_max_numel = 0")
                for param in kernel_input_params:
                    code.writeline(f"_max_numel = max(_max_numel, {param}.size)")
                code.writeline("for shape in out_shapes:")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline("    _max_numel = max(_max_numel, _numel)")

                code.writeline(
                    "_num_tiles = (_max_numel + _tile_size - 1) // _tile_size"
                )

                # Build param names for the kernel
                gmem_input_params = [f"{p}_gmem" for p in kernel_input_params]
                gmem_output_params = [f"{p}_gmem" for p in output_params]
                smem_input_params = [f"{p}_smem" for p in kernel_input_params]
                smem_output_params = [f"{p}_smem" for p in output_params]

                # Generate the TMA kernel with fori_loop
                code.writeline("")
                code.writeline("# Wrapper kernel using lax.fori_loop with direct TMA")

                # Kernel receives: *input_gmem_refs, *output_gmem_refs (from plgpu.kernel)
                # Plus scratch SMEM buffers for inputs and outputs, and barriers for TMA
                wrapper_kernel_params = gmem_input_params + gmem_output_params
                all_smem_params = smem_input_params + smem_output_params
                # Barrier params for TMA operations
                barrier_params = [
                    f"_barrier_{i}" for i in range(len(kernel_input_params))
                ]
                scratch_params = ", ".join(all_smem_params + barrier_params)

                code.writeline(
                    f"def _tma_kernel({', '.join(wrapper_kernel_params)}, *, {scratch_params}):"
                )
                with code.indent():
                    # Define the loop body function
                    code.writeline("")
                    code.writeline("def _tile_body(_tile_idx, _):")
                    with code.indent():
                        code.writeline("_tile_start = _tile_idx * _tile_size")
                        code.writeline("")

                        # TMA load inputs from GMEM to SMEM
                        code.writeline(
                            "# TMA load inputs from GMEM to SMEM (OOB auto-masked)"
                        )
                        for i, (gmem_in, smem_in) in enumerate(
                            zip(gmem_input_params, smem_input_params)
                        ):
                            code.writeline(
                                f"plgpu.copy_gmem_to_smem({gmem_in}.at[pl.ds(_tile_start, _tile_size)], {smem_in}, _barrier_{i})"
                            )

                        # Wait for all input loads
                        code.writeline("")
                        code.writeline("# Wait for TMA loads to complete")
                        for i, _ in enumerate(gmem_input_params):
                            code.writeline(f"plgpu.barrier_wait(_barrier_{i})")

                        # Call the original kernel function with SMEM refs
                        code.writeline("")
                        code.writeline("# Compute on SMEM tiles")
                        kernel_call_args = smem_input_params + smem_output_params
                        kernel_fn = kernel_arg.rstrip(",").strip()
                        code.writeline(f"{kernel_fn}({', '.join(kernel_call_args)})")

                        # TMA store outputs from SMEM to GMEM
                        code.writeline("")
                        code.writeline(
                            "# TMA store outputs from SMEM to GMEM (OOB auto-masked)"
                        )
                        code.writeline("plgpu.commit_smem()")
                        for gmem_out, smem_out in zip(
                            gmem_output_params, smem_output_params
                        ):
                            code.writeline(
                                f"plgpu.copy_smem_to_gmem({smem_out}, {gmem_out}.at[pl.ds(_tile_start, _tile_size)])"
                            )
                        code.writeline("plgpu.wait_smem_to_gmem(0)")
                        code.writeline("")
                        code.writeline("return None")

                    # Run the loop over all tiles
                    code.writeline("")
                    code.writeline("# Iterate over all tiles")
                    code.writeline("lax.fori_loop(0, _num_tiles, _tile_body, None)")

                # Build scratch_shapes dict for SMEM buffers and TMA barriers
                code.writeline("")
                code.writeline(
                    "# Build SMEM scratch shapes for inputs, outputs, and TMA barriers"
                )
                code.writeline("_scratch_shapes = {}")
                for i, smem_param in enumerate(smem_input_params):
                    # Get dtype from input param
                    orig_param = kernel_input_params[i]
                    code.writeline(
                        f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), {orig_param}.dtype)"
                    )
                for i, smem_param in enumerate(smem_output_params):
                    code.writeline(
                        f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), out_dtypes[{i}])"
                    )
                # Add barriers for TMA GMEM->SMEM operations
                for barrier_param in barrier_params:
                    code.writeline(
                        f"_scratch_shapes['{barrier_param}'] = plgpu.Barrier(num_arrivals=1)"
                    )

                # Create flattened and aligned output specs for TMA
                code.writeline("")
                code.writeline("# Create flattened output specs aligned to tile size")
                code.writeline("_flat_out_specs = []")
                code.writeline("for shape, dtype in zip(out_shapes, out_dtypes):")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline(
                    "    _aligned_numel = ((_numel + _tile_size - 1) // _tile_size) * _tile_size"
                )
                code.writeline(
                    "    _flat_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("_flat_out_specs = tuple(_flat_out_specs)")

                # Call plgpu.kernel with the TMA kernel
                code.writeline("")
                code.writeline("# Call plgpu.kernel with TMA kernel")
                code.writeline("_result = plgpu.kernel(")
                with code.indent():
                    code.writeline("_tma_kernel,")
                    code.writeline("out_shape=_flat_out_specs,")
                    code.writeline("scratch_shapes=_scratch_shapes,")
                code.writeline(")(")
                # Pass flattened inputs for 1D tiled processing
                for param in kernel_input_params:
                    code.writeline(f"    {param}.flatten(),")
                code.writeline(")")

                # Reshape outputs to original shapes
                code.writeline("")
                code.writeline("# Reshape results to original shapes")
                code.writeline("if not isinstance(_result, tuple):")
                code.writeline("    _result = (_result,)")
                code.writeline("_final_results = []")
                code.writeline("for _res, _shape in zip(_result, _orig_out_shapes):")
                code.writeline("    _orig_numel = 1")
                code.writeline("    for _s in _shape:")
                code.writeline("        _orig_numel *= _s")
                code.writeline(
                    "    _final_results.append(_res[:_orig_numel].reshape(_shape))"
                )
                code.writeline(
                    "return _final_results[0] if len(_final_results) == 1 else tuple(_final_results)"
                )
            elif self.is_gpu:
                # Legacy GPU path with explicit padding (use_emit_pipeline=False)
                # For GPU, pad inputs to align to WARPGROUP_SIZE (128)
                # Mosaic GPU requires tensor sizes to be multiples of 128
                # BUT: only apply padding when all tensors have the same size
                # (no broadcasting). If inputs have different sizes, we need
                # to preserve shapes for proper broadcasting semantics.

                # First, check if all inputs and outputs have the same numel
                code.writeline(
                    "# Check if all tensors have same size (no broadcasting)"
                )
                code.writeline("_all_sizes = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"_all_sizes.append({param}.size)")
                code.writeline("for shape in out_shapes:")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline("    _all_sizes.append(_numel)")
                code.writeline("_unique_sizes = set(_all_sizes)")
                code.writeline(
                    "_can_pad = len(_unique_sizes) == 1 and all(s > 1 for s in _unique_sizes)"
                )

                code.writeline("")
                code.writeline("if _can_pad:")
                code.writeline("    # All tensors same size - safe to flatten and pad")
                code.writeline("    _orig_out_shapes = out_shapes")
                code.writeline("    _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"    _orig_size_{i} = {param}.size")
                    code.writeline(
                        f"    _aligned_size_{i} = ((_orig_size_{i} + 127) // 128) * 128"
                    )
                    code.writeline(f"    if _orig_size_{i} != _aligned_size_{i}:")
                    code.writeline(f"        _flat_{i} = {param}.flatten()")
                    code.writeline(
                        f"        _padded_{i} = jnp.pad(_flat_{i}, (0, _aligned_size_{i} - _orig_size_{i}))"
                    )
                    code.writeline(f"        _padded_inputs.append(_padded_{i})")
                    code.writeline("    else:")
                    code.writeline(f"        _padded_inputs.append({param}.flatten())")

                code.writeline("    # Align output shapes to warpgroup size (128)")
                code.writeline("    _aligned_out_specs = []")
                code.writeline("    _is_scalar_output = []")
                code.writeline("    for shape, dtype in zip(out_shapes, out_dtypes):")
                code.writeline("        _numel = 1")
                code.writeline("        for s in shape:")
                code.writeline("            _numel *= s")
                code.writeline("        if _numel <= 1:")
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct(shape, dtype))"
                )
                code.writeline("            _is_scalar_output.append(True)")
                code.writeline("        else:")
                code.writeline(
                    "            _aligned_numel = ((_numel + 127) // 128) * 128"
                )
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("            _is_scalar_output.append(False)")
                code.writeline("    _aligned_out_specs = tuple(_aligned_out_specs)")

                code.writeline("    _result = plgpu.kernel(")
                code.writeline("        " + kernel_arg)
                code.writeline("        out_shape=_aligned_out_specs,")
                code.writeline("    )(*_padded_inputs)")

                code.writeline("    # Remove padding from results")
                code.writeline("    if not isinstance(_result, tuple):")
                code.writeline("        _result = (_result,)")
                code.writeline("    _unpadded_results = []")
                code.writeline(
                    "    for _res, _shape, _is_scalar in zip(_result, _orig_out_shapes, _is_scalar_output):"
                )
                code.writeline("        if _is_scalar:")
                code.writeline("            _unpadded_results.append(_res)")
                code.writeline("        else:")
                code.writeline("            _orig_numel = 1")
                code.writeline("            for _s in _shape:")
                code.writeline("                _orig_numel *= _s")
                code.writeline(
                    "            _unpadded = _res[:_orig_numel].reshape(_shape)"
                )
                code.writeline("            _unpadded_results.append(_unpadded)")
                code.writeline(
                    "    return _unpadded_results[0] if len(_unpadded_results) == 1 else tuple(_unpadded_results)"
                )

                code.writeline("else:")
                code.writeline(
                    "    # Different sizes - check if it's a reduction (scalar output)"
                )
                code.writeline("    _out_numel = 1")
                code.writeline("    for s in out_shapes[0]:")
                code.writeline("        _out_numel *= s")
                code.writeline("    ")
                code.writeline("    if _out_numel <= 1:")
                code.writeline(
                    "        # Scalar output (reduction) - pad inputs but keep scalar output"
                )
                code.writeline("        _orig_out_shapes = out_shapes")
                code.writeline("        _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"        _orig_size_{i} = {param}.size")
                    code.writeline(
                        f"        _aligned_size_{i} = ((_orig_size_{i} + 127) // 128) * 128"
                    )
                    code.writeline(f"        if _orig_size_{i} != _aligned_size_{i}:")
                    code.writeline(f"            _flat_{i} = {param}.flatten()")
                    code.writeline(
                        f"            _padded_{i} = jnp.pad(_flat_{i}, (0, _aligned_size_{i} - _orig_size_{i}))"
                    )
                    code.writeline(f"            _padded_inputs.append(_padded_{i})")
                    code.writeline("        else:")
                    code.writeline(
                        f"            _padded_inputs.append({param}.flatten())"
                    )
                code.writeline("        ")
                code.writeline("        # Scalar output - don't pad")
                code.writeline("        _aligned_out_specs = tuple(")
                code.writeline("            jax.ShapeDtypeStruct(shape, dtype)")
                code.writeline(
                    "            for shape, dtype in zip(out_shapes, out_dtypes)"
                )
                code.writeline("        )")
                code.writeline("        ")
                code.writeline("        _result = plgpu.kernel(")
                code.writeline("            " + kernel_arg)
                code.writeline("            out_shape=_aligned_out_specs,")
                code.writeline("        )(*_padded_inputs)")
                code.writeline("        return _result")
                code.writeline("    else:")
                code.writeline(
                    "        # Non-scalar output with broadcasting - broadcast inputs to output shape"
                )
                code.writeline("        _target_shape = out_shapes[0]")
                code.writeline("        _target_numel = _out_numel")
                code.writeline("        _orig_out_shapes = out_shapes")
                code.writeline("        ")
                code.writeline(
                    "        # Broadcast and flatten all inputs to target shape"
                )
                code.writeline("        _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(
                        f"        _broadcasted_{i} = jnp.broadcast_to({param}, _target_shape).flatten()"
                    )
                    code.writeline(
                        f"        _aligned_size_{i} = ((_target_numel + 127) // 128) * 128"
                    )
                    code.writeline(f"        if _target_numel != _aligned_size_{i}:")
                    code.writeline(
                        f"            _padded_{i} = jnp.pad(_broadcasted_{i}, (0, _aligned_size_{i} - _target_numel))"
                    )
                    code.writeline(f"            _padded_inputs.append(_padded_{i})")
                    code.writeline("        else:")
                    code.writeline(
                        f"            _padded_inputs.append(_broadcasted_{i})"
                    )
                code.writeline("        ")
                code.writeline("        # Align output shapes to warpgroup size (128)")
                code.writeline("        _aligned_out_specs = []")
                code.writeline(
                    "        for shape, dtype in zip(out_shapes, out_dtypes):"
                )
                code.writeline("            _numel = 1")
                code.writeline("            for s in shape:")
                code.writeline("                _numel *= s")
                code.writeline(
                    "            _aligned_numel = ((_numel + 127) // 128) * 128"
                )
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("        _aligned_out_specs = tuple(_aligned_out_specs)")
                code.writeline("        ")
                code.writeline("        _result = plgpu.kernel(")
                code.writeline("            " + kernel_arg)
                code.writeline("            out_shape=_aligned_out_specs,")
                code.writeline("        )(*_padded_inputs)")
                code.writeline("        ")
                code.writeline("        # Remove padding from results")
                code.writeline("        if not isinstance(_result, tuple):")
                code.writeline("            _result = (_result,)")
                code.writeline("        _unpadded_results = []")
                code.writeline(
                    "        for _res, _shape in zip(_result, _orig_out_shapes):"
                )
                code.writeline("            _orig_numel = 1")
                code.writeline("            for _s in _shape:")
                code.writeline("                _orig_numel *= _s")
                code.writeline(
                    "            _unpadded = _res[:_orig_numel].reshape(_shape)"
                )
                code.writeline("            _unpadded_results.append(_unpadded)")
                code.writeline(
                    "        return _unpadded_results[0] if len(_unpadded_results) == 1 else tuple(_unpadded_results)"
                )
            else:
                code.writeline("return pl.pallas_call(")
                code.writeline("    " + kernel_arg)
                code.writeline("    out_shape=out_specs,")
                code.writeline(f"    interpret={interpret_literal},")
                code.writeline("    grid=(1,),")
                code.writeline(
                    f"    input_output_aliases={{ {alias_map_literal} }},"
                    if alias_pairs
                    else "    input_output_aliases={},"
                )
                code.writeline(")(")
                if kernel_input_params:
                    code.writeline(f"    {', '.join(kernel_input_params)},")
                code.writeline(")")

        main_name = f"{kernel_name}_main"
        code.writeline(
            f"def {main_name}({', '.join(full_kernel_params)}, stream=None):"
        )
        with code.indent():
            code.writeline("# Enable JAX x64 mode for float64/int64 support")
            code.writeline("jax.config.update('jax_enable_x64', True)")
            # Clear JAX caches to avoid Mosaic GPU backend state issues
            code.writeline("jax.clear_caches()")
            if alias_params:
                code.writeline("# Convert Torch -> JAX for donated outputs")
                for alias_name in alias_params:
                    # TODO: The `jax.device_put` path is a temporary workaround for a Mosaic compiler bug
                    # that occurs with DLPack. Once TorchTPU provides a direct method for placing a
                    # `torch.Tensor` on a TPU device, this should be reverted to use the
                    #  `jax.dlpack.from_dlpack` path.
                    if is_tpu:
                        code.writeline(
                            f"{alias_name}_jax = jax.device_put({alias_name}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        # Explicitly place on CPU to ensure Pallas kernel runs on CPU
                        code.writeline(
                            f"{alias_name}_jax = jax.device_put(jax.dlpack.from_dlpack({alias_name}.detach().contiguous()), device=jax.devices('cpu')[0])"
                        )
            code.writeline("# Convert Torch -> JAX for in-place tensors")
            for ptr in pointer_tail:
                if ptr.startswith("in_out_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        # Explicitly place on CPU to ensure Pallas kernel runs on CPU
                        code.writeline(
                            f"{ptr}_jax = jax.device_put(jax.dlpack.from_dlpack({ptr}.detach().contiguous()), device=jax.devices('cpu')[0])"
                        )
            code.writeline("# Convert Torch -> JAX for inputs")
            for ptr in pointer_tail:
                if ptr.startswith("in_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        # Explicitly place on CPU to ensure Pallas kernel runs on CPU
                        code.writeline(
                            f"{ptr}_jax = jax.device_put(jax.dlpack.from_dlpack({ptr}.detach().contiguous()), device=jax.devices('cpu')[0])"
                        )

            code.writeline("# Prepare output metadata from PyTorch tensor")
            code.writeline(
                "out_shapes = ("
                + ", ".join([f"tuple({name}.shape)" for name in output_params])
                + ",)"
            )
            code.writeline(
                "out_dtypes = ("
                + ", ".join(
                    [
                        f"torch_dtype_to_jax_runtime({name}.dtype)"
                        for name in output_params
                    ]
                )
                + ",)"
            )
            arg_name_map: dict[str, str] = {}
            for alias_name in alias_params:
                arg_name_map[alias_name] = f"{alias_name}_jax"
            for ptr in pointer_tail:
                arg_name_map[ptr] = f"{ptr}_jax"

            # Build the jit_wrapper call with size vars and tensor args
            wrapper_call_args = ["out_shapes", "out_dtypes"]
            # Add size variable params (they're already available as locals in main)
            wrapper_call_args.extend(size_var_params)
            # Add tensor args (with _jax suffix)
            wrapper_call_args.extend(arg_name_map[name] for name in kernel_input_params)
            code.writeline(f"res = {jit_wrapper_name}({', '.join(wrapper_call_args)})")
            if copy_output_indices:
                code.writeline(
                    "result_values = res if isinstance(res, tuple) else (res,)"
                )
                for idx in copy_output_indices:
                    name = output_params[idx]
                    if is_tpu:
                        code.writeline(
                            f"res_cpu = jax.device_get(result_values[{idx}])"
                        )
                        code.writeline(f"{name}.copy_(torch.from_dlpack(res_cpu))")
                    else:
                        code.writeline(
                            f"{name}.copy_(torch.from_dlpack(result_values[{idx}]))"
                        )

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[IRNode] = None) -> None:  # type: ignore[override]
        """Generate the Python code that calls this Pallas kernel."""
        wrapper = V.graph.wrapper_code
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_param_names = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_param_names if p.startswith("out_ptr")]
        call_arg_strs = list(map(str, call_args))
        aliasable = getattr(self, "aliasable_out_ptrs", {})
        alias_call_args = [
            call_arg_strs[kernel_param_names.index(p)]
            for p in pure_out_params
            if aliasable.get(p, False)
        ]

        # Generate kernel call: kernel_name.run(arg1, arg2, ...)
        # Note: async_compile.pallas loads {name}_main function and wraps it in PallasKernelWrapper
        # which exposes a run() method
        kernel_call = f"{name}.run({', '.join(alias_call_args + call_arg_strs)})"
        wrapper.writeline(kernel_call)


class PallasScheduling(SIMDScheduling):
    kernel_type = PallasKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        # Pallas/JAX can handle reductions to single elements efficiently
        # without requiring split reductions
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: PallasKernel,
    ) -> str:  # type: ignore[override]
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"pallas_{kernel_hash}"
        else:
            kernel_name = f"pallas_{fused_name}_{kernel_hash}"
        wrapper.src_to_kernel[src_code] = kernel_name

        # Replace placeholder if any
        src_code = src_code.replace("<KERNEL_NAME>", kernel_name)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.pallas({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
