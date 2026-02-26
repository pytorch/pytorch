from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import typing_extensions
from typing import Any, TYPE_CHECKING

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet
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

    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = None):
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
        return f"jnp.exp({x})"

    @staticmethod
    def exp2(x: str) -> str:
        return f"jnp.exp2({x})"

    @staticmethod
    def expm1(x: str) -> str:
        return f"jnp.expm1({x})"

    @staticmethod
    def log(x: str) -> str:
        return f"jnp.log({x})"

    @staticmethod
    def log10(x: str) -> str:
        return f"jnp.log10({x})"

    @staticmethod
    def log2(x: str) -> str:
        return f"jnp.log2({x})"

    @staticmethod
    def log1p(x: str) -> str:
        return f"jnp.log1p({x})"

    @staticmethod
    def sqrt(x: str) -> str:
        return f"jnp.sqrt({x})"

    @staticmethod
    def rsqrt(x: str) -> str:
        return f"jax.lax.rsqrt({x})"

    @staticmethod
    def abs(x: str) -> str:
        return f"jnp.abs({x})"

    @staticmethod
    def neg(x: str) -> str:
        return f"(-{x})"

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
        return f"jax.nn.sigmoid({x})"

    @staticmethod
    def relu(x: str) -> str:
        return f"jnp.maximum({x}, 0)"

    @staticmethod
    def pow(a: str, b: str) -> str:
        return f"jnp.power({a}, {b})"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        return f"jnp.maximum({a}, {b})"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    @staticmethod
    def where(cond: str, a: str, b: str) -> str:
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
        src_dtype: torch.dtype | None = None,
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

        # Track which iteration variables are used
        V.kernel.used_iter_vars.update(V.kernel._get_used_iter_vars(expr))

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
            return "True" if val else "False"
        # Handle special float values
        if isinstance(val, float):
            if math.isnan(val):
                return "jnp.nan"
            if math.isinf(val):
                return "jnp.inf" if val > 0 else "-jnp.inf"
        return f"jnp.array({val}, dtype={jax_dtype})"

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

    clip = clamp

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

    i0 = modified_bessel_i0

    @staticmethod
    def i0e(x: str) -> str:
        # Exponentially scaled modified Bessel function I_0
        return f"jax.lax.bessel_i0e({x})"

    i1 = modified_bessel_i1

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

    igamma = gammainc

    igammac = gammaincc

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
        return PallasKernelOverrides.chebyshev_polynomial_t(f"(2 * {x} - 1)", n)

    @staticmethod
    def shifted_chebyshev_polynomial_u(x: str, n: str) -> str:
        return PallasKernelOverrides.chebyshev_polynomial_u(f"(2 * {x} - 1)", n)

    @staticmethod
    def shifted_chebyshev_polynomial_v(x: str, n: str) -> str:
        return PallasKernelOverrides.chebyshev_polynomial_v(f"(2 * {x} - 1)", n)

    @staticmethod
    def shifted_chebyshev_polynomial_w(x: str, n: str) -> str:
        return PallasKernelOverrides.chebyshev_polynomial_w(f"(2 * {x} - 1)", n)

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


@dataclasses.dataclass
class _CodegenContext:
    """Bundles local state shared across codegen_kernel helper methods."""

    code: IndentedBuffer
    kernel_name: str
    is_tpu: bool
    interpret_is_cpu: bool
    interpret_literal: str
    kernel_params: list[str]
    pure_out_params: list[str]
    output_params: list[str]
    size_var_params: list[str]
    output_buffer_lookup: dict[str, str]
    aliasable_flags: dict[str, bool]
    alias_params: list[str]
    pointer_tail: list[str]
    kernel_input_params: list[str]
    full_kernel_params: list[str]
    non_alias_out_set: OrderedSet[str]
    copy_output_indices: list[int]
    alias_pairs: list[tuple[int, int]]


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
        # Track which iteration variables are actually used in the kernel
        self.used_iter_vars: OrderedSet[sympy.Symbol] = OrderedSet()
        # Track if any load/store uses flatten-based indexing (buf[...].flatten()[idx])
        self.has_flatten_indexing = False
        # Track input buffers that are accessed with transposed last-2 dims
        self.transposed_input_buffers: OrderedSet[str] = OrderedSet()

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
            # Track which iteration variables are used before returning
            self.used_iter_vars.update(self._get_used_iter_vars(index))
            # Generate actual index expression - iteration variables are already
            # defined as jnp.arange arrays, so we just convert to JAX code
            return self.kexpr(index)

        # Simplify the index
        index = V.graph.sizevars.simplify(index)
        # Find which iteration variable(s) are used
        used_vars = self._get_used_iter_vars(index)

        # Track which iteration variables are used
        self.used_iter_vars.update(used_vars)

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

        # Track which iteration variables are used
        self.used_iter_vars.update(used_vars)

        # Convert sympy expression to Python/JAX code string
        # The iteration variables are already defined as jnp.arange arrays
        index_str = self.kexpr(index)

        # Mark this as requiring flatten access
        return index_str

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

    def _is_transposed_access(self, name: str, index: sympy.Expr) -> bool:
        """Check if buffer access needs transpose.

        Transpose on load is needed when:
        1. Non-square buffers: dimensions are swapped relative to iteration vars
        2. Square buffers: index coefficient pattern indicates transposed access
           (first iteration var has larger coefficient than second)
        """
        info = self._get_buffer_info(name)
        if info is None:
            return False

        _, buf_size, _, actual_strides, _ = info

        # Only handle 2D buffers
        if len(buf_size) != 2 or len(actual_strides) != 2:
            return False

        size0 = self._safe_int(buf_size[0])
        size1 = self._safe_int(buf_size[1])
        if size0 is None or size1 is None or size0 <= 1 or size1 <= 1:
            return False

        s0 = actual_strides[0]
        s1 = actual_strides[1]
        if s0 is None or s1 is None:
            return False

        # Get iteration variable info
        var_items = list(self.range_tree_nodes.items())
        if len(var_items) < 2:
            return False

        # Skip for reduction variables

        if any(entry.is_reduction for _, entry in var_items):
            return False

        # Extract coefficients from index expression
        inner_var = var_items[0][0]
        outer_var = var_items[1][0]
        index = V.graph.sizevars.simplify(index)

        inner_coeff = self._get_index_coefficient(index, inner_var)
        outer_coeff = self._get_index_coefficient(index, outer_var)

        if inner_coeff != 0 and outer_coeff != 0:
            # Only transpose for standard row-major buffers (stride[0] = size[1], stride[1] = 1)
            is_standard_row_major = s0 == size1 and s1 == 1
            if not is_standard_row_major:
                return False

            # Only transpose if output is column-major (indicates actual transpose op)
            output_is_column_major = self._has_column_major_output()
            if not output_is_column_major:
                return False

            # Check if coefficients indicate transposed access
            inner_matches_s0 = abs(inner_coeff - s0) < abs(inner_coeff - s1)
            outer_matches_s1 = abs(outer_coeff - s1) < abs(outer_coeff - s0)
            return inner_matches_s0 and outer_matches_s1

        return False

    def _has_column_major_output(self) -> bool:
        """Check if any output buffer has column-major stride layout."""
        output_buffers = getattr(self.args, "output_buffers", {})
        # Check both explicit output buffers and graph buffers (which may not
        # be populated during load).
        buf_names = itertools.chain(output_buffers, V.graph.name_to_buffer)
        for buf_name in buf_names:
            out_buf = V.graph.get_buffer(buf_name)
            if out_buf is None:
                continue
            if buf_name not in output_buffers and not isinstance(
                out_buf, ComputedBuffer
            ):
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
    def _safe_int(val: Any) -> int | None:
        """Convert value to int, returning None on failure."""
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _compute_prefix_numel(self, prefixes: OrderedSet) -> int | None:
        """Compute total numel for given prefixes (e.g., pointwise prefixes)."""
        result = 1
        for p in prefixes:
            if p in self.numels:
                numel = self._safe_int(self.numels[p])
                if numel is None:
                    return None
                result *= numel
        return result

    def _compute_reduction_numel(self) -> int | None:
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
            info = self._get_buffer_info(name)
            if info is None:
                return False
            buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = info
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

    def _get_buffer_info(self, name: str) -> tuple[Any, Any, Any, list, bool] | None:
        """Get buffer metadata (buf_obj, buf_size, buf_numel, actual_strides, is_contiguous).

        Returns None if the buffer doesn't exist.
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return None
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

        info = self._get_buffer_info(name)
        if info is None:
            return index_str, needs_flatten

        buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = info
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
        is_tpu = V.graph.get_current_device_or_throw().type == "tpu"
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

    def _try_multidim_slice(
        self,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> tuple[str, bool]:
        """
        Try to emit multi-dim slice notation instead of flatten + gather.

        For a buffer with shape (d0, ..., dk) and index `stride * var + offset`,
        emit `buf[:, ..., :, offset::stride]` when stride divides dk.
        """
        if not needs_flatten:
            return index_str, needs_flatten

        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return index_str, needs_flatten

        buf_size = buf_obj.get_size()
        ndim = len(buf_size)
        if ndim < 2:
            return index_str, needs_flatten

        # Need a single iteration variable with an affine index
        used_vars = self._get_used_iter_vars(index)
        if len(used_vars) != 1:
            return index_str, needs_flatten

        var = next(iter(used_vars))
        var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)
        stride = self._safe_int(
            BlockPatternMatcher.match_affine_block_expr(var_expr, var)
        )
        if stride is None or stride <= 1:
            return index_str, needs_flatten

        offset = V.graph.sizevars.simplify(index - var_expr)
        try:
            offset_val = int(offset)
        except (TypeError, ValueError):
            return index_str, needs_flatten

        if offset_val < 0 or offset_val >= stride:
            return index_str, needs_flatten

        last_dim = self._safe_int(buf_size[-1])
        if last_dim is None or last_dim % stride != 0:
            return index_str, needs_flatten

        # Verify the iteration variable covers all buffer elements at the
        # given stride: var_length * stride == buf_numel. This ensures
        # the flattened stride-access 0, stride, 2*stride, ... maps exactly
        # to buf[:, ..., :, offset::stride].
        entry = self.range_tree_nodes.get(var)
        if entry is None:
            return index_str, needs_flatten
        var_length = self._safe_int(entry.length)
        buf_numel = 1
        for s in buf_size:
            d = self._safe_int(s)
            if d is None:
                return index_str, needs_flatten
            buf_numel *= d
        if var_length is None or var_length * stride != buf_numel:
            return index_str, needs_flatten

        prefix = ":, " * (ndim - 1)
        if offset_val == 0:
            slice_str = f"{prefix}::{stride}"
        else:
            slice_str = f"{prefix}{offset_val}::{stride}"
        return slice_str, False

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
            self.has_flatten_indexing = True
            # Flatten then index for non-contiguous access (gather operation)
            has_minmax = index.has(sympy.Min) or index.has(sympy.Max)
            idx = f"({index_str}).astype(jnp.int64)" if has_minmax else index_str
            return f"{buf}[...].flatten()[{idx}]"
        else:
            # Direct indexing for contiguous access
            load_expr = f"{buf}[{index_str}]"

            # Check for transposed access
            if index_str == "..." and self._is_transposed_access(name, index):
                load_expr = f"jnp.transpose({load_expr})"
                self.has_transposed_load = True

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

        info = self._get_buffer_info(name)
        if info is None:
            return False

        _, buf_size, _, actual_strides, _ = info
        if len(actual_strides) != 2 or len(buf_size) != 2:
            return False

        size0 = self._safe_int(buf_size[0])
        size1 = self._safe_int(buf_size[1])
        s0 = actual_strides[0]
        s1 = actual_strides[1]

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
            inp_info = self._get_buffer_info(inp_name)
            if inp_info is None:
                continue
            _, _, _, inp_strides, _ = inp_info
            if len(inp_strides) != 2:
                continue
            inp_s0 = inp_strides[0]
            inp_s1 = inp_strides[1]
            if inp_s0 is not None and inp_s1 is not None and inp_s0 < inp_s1:
                return False  # Input is also column-major

        return True

    def _build_full_array_store_expr(
        self, out: str, value: CSEVariable, needs_transpose: bool
    ) -> list[str]:
        """
        Build store expression for full array assignment.

        Handles scalar broadcast, shape matching, and optional transpose.
        Returns a list of lines to emit (variable assignment + store).
        """
        lines = [f"_val = jnp.asarray({value})"]
        if needs_transpose:
            lines.append(
                f"{out}[...] = "
                f"jnp.full({out}.shape, _val) if _val.ndim == 0 "
                f"else jnp.transpose(_val)"
            )
        else:
            lines.append(
                f"{out}[...] = "
                f"jnp.full({out}.shape, _val) if _val.ndim == 0 "
                f"else (_val.reshape({out}.shape) if _val.size == {out}.size "
                f"else jnp.broadcast_to(_val, {out}.shape))"
            )
        return lines

    def _build_store_expr(
        self,
        out: str,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        index_str: str,
        needs_flatten: bool,
        mode: Any = None,
    ) -> list[str]:
        """
        Build the store expression based on indexing mode.
        mode can be None (set) or "atomic_add" (accumulate).
        Returns a list of lines to emit.
        """
        if index_str == "...":
            # Full array store with shape matching
            needs_transpose = self._check_store_needs_transpose(name)
            return self._build_full_array_store_expr(out, value, needs_transpose)

        if needs_flatten:
            self.has_flatten_indexing = True
            # Block variable indexing (e.g., im2col) - use flattened scatter
            scatter_op = "add" if mode == "atomic_add" else "set"
            return [
                f"{out}[...] = {out}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                f"jnp.asarray({value}).flatten()).reshape({out}.shape)"
            ]

        # Direct indexed assignment
        has_indirect = self._has_indirect_vars(index)
        buf = V.graph.get_buffer(name)

        if buf is not None:
            buf_size = buf.get_size()
            if len(buf_size) > 1 and not self._has_iteration_vars(index):
                # Multi-dim output with constant index - use [...] for full assignment
                return self._build_full_array_store_expr(out, value, False)

        if has_indirect:
            # Indirect indexed store (scatter): use .add() for atomic_add, .set() otherwise
            scatter_op = "add" if mode == "atomic_add" else "set"
            lines = [f"_val = jnp.asarray({value})"]
            value_expr = (
                f"(jnp.full({index_str}.shape, _val) if _val.ndim == 0 else {value})"
            )
            if mode == "atomic_add":
                # For atomic_add, mark output as needing to be readable (for aliasing)
                self.outputs_need_read.add(out)
                alias_param = f"{out}_alias"
                lines.append(
                    f"{out}[...] = {alias_param}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                    f"{value_expr}.flatten()).reshape({out}.shape)"
                )
            else:
                lines.append(f"{out}[{index_str}] = {value_expr}")
            return lines

        return [f"{out}[{index_str}] = {value}"]

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
            index_parts.append(indirect_var)
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

        # Try to emit multi-dim slice instead of flatten + gather
        index_str, needs_flatten = self._try_multidim_slice(
            name, index, index_str, needs_flatten
        )

        # Build the load expression
        load_expr = self._build_load_expr(buf, name, index, index_str, needs_flatten)

        # Handle intermediate buffer squeezing for correct broadcasting
        if not needs_flatten and index_str == "...":
            load_expr = self._maybe_squeeze_intermediate_buffer(name, load_expr)
            # Handle 1D buffer broadcasting for higher-dimensional kernels
            load_expr = self._maybe_broadcast_1d_buffer(name, index, load_expr)

        return self.cse.generate(
            self.compute,
            load_expr,
            dtype=dtype,
        )

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

        # Track which iteration variables are used
        self.used_iter_vars.update(used_iter_vars_set)

        if len(used_iter_vars_set) == 0:
            return self.kexpr(index)

        # Sort iteration variables by their coefficient (stride) in the index expression.
        # Variables with larger strides correspond to earlier output dimensions.
        # Use inf default so symbolic coefficients sort as outermost dimensions.
        def _coeff(var):
            return self._get_index_coefficient(index, var, default=float("inf"))

        used_iter_vars = sorted(used_iter_vars_set, key=_coeff, reverse=True)
        iter_coeffs = [_coeff(var) for var in used_iter_vars]

        # Rename symbolic sizes to kernel parameter names
        index_str = self.kexpr(self.rename_indexing(index))
        indirect_var_syms = self._get_indirect_vars(index)
        indirect_vars = [str(sym) for sym in indirect_var_syms]

        # Get coefficients for indirect vars to determine output ordering
        indirect_coeffs = {str(s): _coeff(s) for s in indirect_var_syms}

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
            all_components.append((_coeff(var), "iter", var))
        for sym in indirect_var_syms:
            all_components.append((_coeff(sym), "indirect", sym))
        all_components.sort(key=lambda x: x[0], reverse=True)

        # Calculate trailing dims needed for each component
        # Each component needs trailing dims for all subsequent iter vars
        # plus trailing dims for all dimensions of subsequent indirect vars
        # For simplicity, assume each indirect var contributes some dimensions
        # that will be handled by the reshape at store time

        # For iter vars, we need to count how many dimensions come after in the output
        for i, var in enumerate(used_iter_vars):
            var_name = str(var)
            if var in self.range_tree_nodes:
                range_entry = self.range_tree_nodes[var]
                range_size = range_entry.length
                # Rename to use kernel parameter names for symbolic sizes
                renamed_size = self.rename_indexing(range_size)
                var_coeff = _coeff(var)

                arange_expr = f"jnp.arange({self.kexpr(renamed_size)})"

                # Count trailing dims needed:
                # - One for each subsequent iter var (with smaller coeff)
                # - One for each dimension of indirect vars with smaller coeff
                # For indirect vars, assume each contributes 2 dims (common case)
                # The actual reshape at store time will fix any shape mismatches
                n_trailing_iter = sum(1 for c in iter_coeffs if c < var_coeff)
                n_trailing_indirect = sum(
                    2 for c in indirect_coeffs.values() if c < var_coeff
                )
                n_trailing = n_trailing_iter + n_trailing_indirect

                if n_trailing > 0:
                    trailing_dims = ", None" * n_trailing
                    arange_expr = f"{arange_expr}[:{trailing_dims}]"

                index_str = index_str.replace(var_name, arange_expr)

        # Reshape indirect variables for proper broadcasting.
        for indirect_var in indirect_vars:
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
            store_lines = [
                f"_val = jnp.asarray({value})",
                f"{out}[...] = jnp.full({out}.shape, _val) if _val.ndim == 0 else _val.reshape({out}.shape)",
            ]
        else:
            # Check for scatter pattern (indirect indexing for stores)
            scatter_info = self._detect_scatter_pattern(index, name)

            if scatter_info is not None:
                # Track iteration variables used in scatter index
                self.used_iter_vars.update(self._get_used_iter_vars(index))
                store_lines = [
                    self._build_scatter_store_expr(out, value, scatter_info, name, mode)
                ]
            else:
                # Get base index expression
                index_str, needs_flatten = self._get_index_expr(index)

                # Check for im2col-like patterns
                index_str, needs_flatten = self._check_im2col_pattern(
                    index, index_str, needs_flatten
                )

                # Build the store expression
                store_lines = self._build_store_expr(
                    out, name, index, value, index_str, needs_flatten, mode
                )

        for line in store_lines:
            self.stores.writeline(line)
            # Track which output param this store uses for filtering in codegen_kernel
            self.store_with_output.append((out, line))

    @staticmethod
    def _get_index_coefficient(
        index: sympy.Expr, var: sympy.Symbol, default: int | float = 0
    ) -> int | float:
        """Get integer coefficient of a variable in an index expression."""
        coeff = index.coeff(var)
        if coeff == 0:
            coeff = sympy.diff(index, var)
        try:
            return int(coeff)
        except (TypeError, ValueError):
            return default

    def _detect_scatter_pattern(
        self, index: sympy.Expr, output_name: str = ""
    ) -> dict[str, Any] | None:
        """Detect scatter operation pattern. Returns scatter info dict or None."""
        indirect_syms = self._get_indirect_vars(index)
        if len(indirect_syms) != 1:
            return None

        indirect_sym = indirect_syms[0]
        indirect_var = str(indirect_sym)
        indirect_coeff: int = int(self._get_index_coefficient(index, indirect_sym))
        if indirect_coeff == 0:
            return None

        # Point scatter: no iteration variables, just indirect indexing
        if not self._has_iteration_vars(index):
            return self._detect_point_scatter(output_name, indirect_var, indirect_coeff)

        # Regular scatter: has both indirect and iteration variables
        return self._detect_iter_scatter(index, indirect_var, indirect_coeff)

    def _detect_point_scatter(
        self, output_name: str, indirect_var: str, indirect_coeff: int
    ) -> dict[str, Any] | None:
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
    ) -> dict[str, Any] | None:
        """Detect scatter pattern with iteration variables."""
        used_iter_vars = self._get_used_iter_vars(index)

        # Collect (var_name, coefficient, length) for each variable
        all_vars: list[tuple[str, int, int]] = []
        for var in used_iter_vars:
            coeff = int(self._get_index_coefficient(index, var))
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
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:  # type: ignore[override]
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
        pointwise_numel: int | None = self._compute_prefix_numel(pointwise_prefixes)
        reduction_numel: int | None = self._compute_reduction_numel()

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
                reduction_expr = f"pallas_partial_reduce({reduction_op}, {value}, {pointwise_numel}, {reduction_numel})"
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

        # Cache the result
        self.cse.reduction_cache[cache_key] = result
        return result

    @staticmethod
    def _buffer_is_contiguous(buffer_name: str) -> bool:
        buf = V.graph.get_buffer(buffer_name)
        layout = buf.get_layout()
        return layout.is_contiguous()

    def _can_tile_cpu_tpu(self) -> bool:
        """Check if this kernel can use tiling on CPU/TPU.

        Tiling is compatible with reductions, transpositions, and multi-range-tree
        kernels as long as no flatten-based indexing is used (buf[...].flatten()[idx]).
        Flatten indexing requires global flat indices which don't decompose into
        per-tile local indices.

        Reject:
        - GPU (has its own TMA / padding path)
        - Flatten-based indexing
        - Scatter outputs (indirect indexing complicates tile boundaries)
        """
        if self.is_gpu:
            return False
        if self.has_flatten_indexing:
            return False
        if self.outputs_need_read:
            return False

        # If iteration variables appear in the compute body (not just in
        # load/store index resolution that collapses to [...]), tiling is
        # unsafe because the arange-based vars have full-tensor shapes.
        if self.used_iter_vars:
            compute_text = "\n".join(str(line) for line in self.compute._lines)
            for var_sym in self.used_iter_vars:
                if str(var_sym) in compute_text:
                    return False

        # Determine the reference output shape (highest-ndim output).
        out_bufs = list(self.args.output_buffers.keys())

        # Only check the current kernel's actual output buffers for transpose,
        # not _has_column_major_output() which scans all graph buffers and can
        # be triggered by unrelated intermediates (e.g., (N,1) reductions with
        # degenerate column-major strides).
        has_col_major_out = False
        for buf_name in out_bufs:
            info = self._get_buffer_info(buf_name)
            if info is None:
                continue
            _, buf_size, _, actual_strides, _ = info
            if len(actual_strides) >= 2 and len(buf_size) >= 2:
                s0 = actual_strides[0]
                s1 = actual_strides[1]
                d0 = self._safe_int(buf_size[0])
                d1 = self._safe_int(buf_size[1])
                if (
                    s0 is not None
                    and s1 is not None
                    and s0 < s1
                    and d0 is not None
                    and d1 is not None
                    and d0 > 1
                    and d1 > 1
                ):
                    has_col_major_out = True
                    break
        self.tile_has_transpose = self.has_transposed_load or has_col_major_out

        # Count trailing reduction dimensions in the output shape that must
        # not be tiled (the kernel body needs the full reduction range).
        # Only count when the kernel actually performs reduction (numel > 1).
        reduction_numel = self._compute_reduction_numel()
        has_reduction = reduction_numel is not None and reduction_numel > 1
        self.tile_skip_last_n = (
            sum(1 for tree in self.range_trees if tree.is_reduction)
            if has_reduction
            else 0
        )

        ref_shape: list[int] = []
        for buf_name in out_bufs:
            info = self._get_buffer_info(buf_name)
            if info is None:
                return False
            _, buf_size, _, _, _ = info
            int_size = [self._safe_int(s) for s in buf_size]
            if any(s is None for s in int_size):
                return False
            if len(int_size) > len(ref_shape):
                ref_shape = int_size  # type: ignore[assignment]

        if not ref_shape:
            return False
        ref_nd = len(ref_shape)

        all_bufs = list(self.args.input_buffers) + out_bufs
        has_tileable = False
        for buf_name in all_bufs:
            info = self._get_buffer_info(buf_name)
            if info is None:
                return False
            _, buf_size, _, _, _ = info
            if len(buf_size) == 0:
                continue  # scalar
            int_size = [self._safe_int(s) for s in buf_size]
            if any(s is None for s in int_size):
                return False
            buf_nd = len(int_size)

            if buf_nd == ref_nd:
                # Same ndim: check dimensions match or are broadcast (1).
                # Allow transposed last-2 dims.
                mismatch = False
                for i in range(ref_nd):
                    if (
                        int_size[i] == ref_shape[i]
                        or int_size[i] == 1
                        or ref_shape[i] == 1
                    ):
                        continue
                    mismatch = True
                    break

                if mismatch and ref_nd >= 2 and self.tile_has_transpose:
                    # Check if last-2 dims are swapped (transpose pattern).
                    # Only allow when the kernel actually transposes
                    # (has_transposed_load or column-major output).
                    if (
                        int_size[-2] == ref_shape[-1]
                        and int_size[-1] == ref_shape[-2]
                        and all(
                            int_size[i] == ref_shape[i]
                            or int_size[i] == 1
                            or ref_shape[i] == 1
                            for i in range(ref_nd - 2)
                        )
                    ):
                        if buf_name in self.args.input_buffers:
                            self.transposed_input_buffers.add(buf_name)
                    else:
                        return False
                elif mismatch:
                    return False

                # At least one buffer with a tileable dim
                if any(
                    int_size[i] == ref_shape[i] and ref_shape[i] > 1
                    for i in range(ref_nd)
                ):
                    has_tileable = True

            elif buf_nd > ref_nd:
                # Reduction input with extra dims. Find an alignment offset k
                # such that buf_shape[k+i] == ref_shape[i] for all i (skipping
                # broadcast dims where ref_shape[i] == 1).
                found = False
                for k in range(buf_nd - ref_nd + 1):
                    ok = True
                    for i in range(ref_nd):
                        if ref_shape[i] == 1:
                            continue
                        if int_size[k + i] != ref_shape[i]:
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if not found:
                    return False
                has_tileable = True

            else:
                # Fewer dims: verify numpy-style broadcastability
                for a, b in zip(reversed(int_size), reversed(ref_shape)):
                    if a != b and a != 1 and b != 1:
                        return False

        if not has_tileable:
            return False

        # On CPU (interpret mode) each tile iteration has significant
        # Python/JAX overhead, so cap the grid size to avoid regressions
        # on large tensors.  On TPU the grid executes natively.
        is_tpu = V.graph.get_current_device_or_throw().type == "tpu"
        if not is_tpu:
            from ..runtime.runtime_utils import pallas_compute_tiling

            _, grid, _ = pallas_compute_tiling(
                tuple(ref_shape),
                transpose=self.tile_has_transpose,
                skip_last_n=self.tile_skip_last_n,
                exact_only=True,
            )
            _MAX_GRID_PRODUCT = 64
            grid_product = 1
            for g in grid:
                grid_product *= g
            if grid_product > _MAX_GRID_PRODUCT:
                return False

        return True

    def codegen_kernel(self, name: str | None = None) -> str:  # type: ignore[override]
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
        is_tpu = V.graph.get_current_device_or_throw().type == "tpu"
        interpret_is_cpu = V.graph.get_current_device_or_throw().type == "cpu"
        interpret_literal = "True" if interpret_is_cpu else "False"

        aliasable_flags: dict[str, bool] = {}
        for param in pure_out_params:
            aliasable_flags[param] = True
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
        # On CPU (interpret=True), pallas_call returns new arrays so we must
        # copy back every output.  On TPU, call_custom_kernel with
        # input_output_aliases handles donation (zero-copy), so no copy is
        # needed.  On CUDA, aliased outputs are mutated in-place by the
        # donated-buffer mechanism so only non-aliased outputs need a copy.
        if interpret_is_cpu:
            copy_output_indices = list(range(len(output_params)))
        elif is_tpu:
            copy_output_indices = []
        else:
            copy_output_indices = [
                idx
                for idx, name in enumerate(output_params)
                if name in non_alias_out_set
            ]

        ctx = _CodegenContext(
            code=code,
            kernel_name=kernel_name,
            is_tpu=is_tpu,
            interpret_is_cpu=interpret_is_cpu,
            interpret_literal=interpret_literal,
            kernel_params=kernel_params,
            pure_out_params=pure_out_params,
            output_params=output_params,
            size_var_params=size_var_params,
            output_buffer_lookup=output_buffer_lookup,
            aliasable_flags=aliasable_flags,
            alias_params=alias_params,
            pointer_tail=pointer_tail,
            kernel_input_params=kernel_input_params,
            full_kernel_params=full_kernel_params,
            non_alias_out_set=non_alias_out_set,
            copy_output_indices=copy_output_indices,
            alias_pairs=[],
        )
        self.aliasable_out_ptrs = aliasable_flags

        self._codegen_imports(ctx)

        # Generate kernel body into a separate buffer first.
        # This allows us to discover all size variables (registered via rename_indexing)
        # before generating the kernel signature.
        kernel_body = IndentedBuffer()
        with kernel_body.indent():
            self._codegen_iteration_vars(kernel_body, ctx)

            for line in self.compute._lines:
                kernel_body.writeline(str(line))

        # Recompute kernel parameters after kernel body generation.
        # Size variables may have been registered during kernel body generation
        # (e.g., via rename_indexing for symbolic sizes), so we need to re-fetch
        # the arg defs to capture all parameters including newly-registered size vars.
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        size_var_names = OrderedSet(self.args.sizevars.values())
        ctx.size_var_params = [p for p in kernel_params if p in size_var_names]
        ctx.pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        ctx.kernel_input_params = alias_params + ctx.pointer_tail
        ctx.full_kernel_params = alias_params + kernel_params

        # Decide whether to use tiling for CPU/TPU after kernel body is fully
        # generated (used_iter_vars is populated during load/store codegen).
        self.tile_cpu_tpu = self._can_tile_cpu_tpu()

        # Emit the kernel function with the correct signature
        kernel_signature = (
            f"def {kernel_name}_kernel({', '.join(ctx.full_kernel_params)}):"
        )
        code.writeline(kernel_signature)

        with code.indent():
            for line in kernel_body._lines:
                if isinstance(line, str):
                    code.writeline(line.lstrip())
                else:
                    code._lines.append(line)

            # Filter stores to only emit those for outputs that are in kernel params.
            for out_ptr, store_line in self.store_with_output:
                if out_ptr in ctx.full_kernel_params:
                    code.writeline(store_line)

        code.writeline("")
        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        donate_indices = []
        base_offset = 2 + len(ctx.size_var_params)
        for idx, name in enumerate(ctx.kernel_input_params):
            if (name in alias_params) or name.startswith("in_out_ptr"):
                donate_indices.append(idx + base_offset)
        if donate_indices:
            donate_literal = "(" + ", ".join(str(x) for x in donate_indices) + ",)"
        else:
            donate_literal = "()"
        static_argnums = list(range(2 + len(ctx.size_var_params)))
        static_argnums_literal = "(" + ", ".join(str(x) for x in static_argnums) + ",)"
        code.writeline(
            "@functools.partial("
            f"jax.jit, static_argnums={static_argnums_literal}, donate_argnums="
            f"{donate_literal})"
        )
        wrapper_params = (
            ["out_shapes", "out_dtypes"] + ctx.size_var_params + ctx.kernel_input_params
        )
        code.writeline(f"def {jit_wrapper_name}({', '.join(wrapper_params)}):")

        alias_pairs: list[tuple[int, int]] = []
        for out_idx, name in enumerate(ctx.output_params):
            if name.startswith("out_ptr"):
                if aliasable_flags.get(name, False):
                    alias_name = f"{name}_alias"
                    input_idx = ctx.kernel_input_params.index(alias_name)
                    alias_pairs.append((input_idx, out_idx))
            else:
                input_idx = ctx.kernel_input_params.index(name)
                alias_pairs.append((input_idx, out_idx))
        alias_map_literal = ", ".join(f"{i}: {o}" for (i, o) in alias_pairs)
        ctx.alias_pairs = alias_pairs

        with code.indent():
            # Pallas requires >= 1-d tensors; promote 0-d to (1,)
            code.writeline(
                "_pallas_out_shapes = tuple("
                "s if len(s) > 0 else (1,) for s in out_shapes)"
            )
            # Reshape aliased inputs to match promoted output shapes
            for input_idx, out_idx in alias_pairs:
                param = ctx.kernel_input_params[input_idx]
                code.writeline(
                    f"{param} = {param}.reshape(_pallas_out_shapes[{out_idx}])"
                )
            code.writeline("out_shapes_pallas = tuple(")
            code.writeline("    jax.ShapeDtypeStruct(shape, dtype)")
            code.writeline(
                "    for shape, dtype in zip(_pallas_out_shapes, out_dtypes)"
            )
            code.writeline(")")
            if self.tile_cpu_tpu:
                self._codegen_tiled_specs(ctx)
            else:
                code.writeline("indexer = lambda n: lambda i: [jnp.int32(i)] * n")
                code.writeline("out_specs_pallas = tuple(")
                code.writeline("    pl.BlockSpec(shape, indexer(len(shape)))")
                code.writeline(
                    "    for shape, dtype in zip(_pallas_out_shapes, out_dtypes)"
                )
                code.writeline(")")
                code.writeline("in_specs_pallas = tuple(")
                code.writeline("    pl.BlockSpec(i.shape, indexer(len(i.shape)))")
                code.writeline(
                    "    for i in [" + ", ".join(ctx.kernel_input_params) + "]"
                )
                code.writeline(")")

            # Wrap kernel with functools.partial to pass scalar arguments (size variables)
            partial_args = []
            for sv_param in ctx.size_var_params:
                partial_args.append(f"{sv_param}={sv_param}")

            if partial_args:
                kernel_arg = f"functools.partial({kernel_name}_kernel, {', '.join(partial_args)}),"
            else:
                kernel_arg = f"{kernel_name}_kernel,"

            use_tma = (
                self.is_gpu and self.use_emit_pipeline and self._can_use_tma_approach()
            )
            if use_tma:
                self._codegen_jit_wrapper_tma(ctx, kernel_arg)
            elif self.is_gpu:
                self._codegen_jit_wrapper_legacy_gpu(ctx, kernel_arg)
            else:
                self._codegen_jit_wrapper_cpu_tpu(
                    ctx, kernel_arg, alias_pairs, alias_map_literal
                )

        self._codegen_main_entry(ctx, jit_wrapper_name)
        return code.getvalue()

    def _codegen_imports(self, ctx: _CodegenContext) -> None:
        imports = """
import functools
import math
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from torch._inductor.runtime.runtime_utils import (
    pallas_compute_tiling, pallas_make_block_spec,
    pallas_gpu_align_output_specs, pallas_gpu_pad_inputs,
    pallas_gpu_unpad_results, pallas_partial_reduce,
    torch_dtype_to_jax_runtime,
)
"""
        if ctx.is_tpu:
            imports += "\nimport jax.export"
            imports += "\nfrom torch_tpu._internal.pallas import tpu_torch_pallas"
        elif not ctx.interpret_is_cpu:
            imports += "\nfrom jax.experimental.pallas import mosaic_gpu as plgpu"
        ctx.code.splice(imports, strip=True)

    def _codegen_iteration_vars(
        self, kernel_body: IndentedBuffer, ctx: _CodegenContext
    ) -> None:
        # Generate iteration variables as jnp.arange arrays
        # Skip on GPU - jnp.arange is not supported by Pallas Mosaic backend
        if not (self.range_tree_nodes and not self.is_gpu and self.used_iter_vars):
            return

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

        candidate_buf_names = []
        if ctx.output_params:
            buf_name = ctx.output_buffer_lookup.get(ctx.output_params[0])
            if buf_name:
                candidate_buf_names.append(buf_name)
        candidate_buf_names.extend(self.args.input_buffers)

        reshape_target_shape, reshape_target_numel = None, None
        for buf_name in candidate_buf_names:
            result = _get_nd_shape_if_matches(buf_name)
            if result[0]:
                reshape_target_shape, reshape_target_numel = result
                break

        var_items = list(self.range_tree_nodes.items())

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
            if var_sym not in self.used_iter_vars:
                continue
            var_name = str(var_sym)
            length = entry.length
            renamed_length = self.rename_indexing(length)
            length_str = self.kexpr(renamed_length)
            length_val = self._safe_int(length)

            if length_val is None:
                if (
                    reshape_target_shape
                    and num_broadcast_dims > 1
                    and idx != total_var_idx
                ):
                    broadcast_idx = next(
                        (
                            i
                            for i, (vidx, _, _, _) in enumerate(broadcast_vars)
                            if vidx == idx
                        ),
                        None,
                    )
                    if broadcast_idx is not None:
                        axis_idx = self._broadcast_axis_idx(
                            broadcast_vars, broadcast_idx, num_broadcast_dims
                        )
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
                shape_str = ", ".join(str(s) for s in reshape_target_shape)
                arange = f"jnp.arange({length_str})"
                kernel_body.writeline(f"{var_name} = {arange}.reshape({shape_str})")
            elif num_broadcast_dims > 1 and idx != total_var_idx:
                broadcast_idx = next(
                    i for i, (vidx, _, _, _) in enumerate(broadcast_vars) if vidx == idx
                )
                axis_idx = self._broadcast_axis_idx(
                    broadcast_vars, broadcast_idx, num_broadcast_dims
                )
                shape_parts = ["1"] * num_broadcast_dims
                shape_parts[axis_idx] = length_str
                shape_str = ", ".join(shape_parts)
                arange = f"jnp.arange({length_str})"
                kernel_body.writeline(f"{var_name} = {arange}.reshape({shape_str})")
            else:
                kernel_body.writeline(f"{var_name} = jnp.arange({length_str})")

    @staticmethod
    def _broadcast_axis_idx(
        broadcast_vars: list[tuple[int, Any, Any, Any]],
        broadcast_idx: int,
        num_broadcast_dims: int,
    ) -> int:
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
            return broadcast_idx
        return num_broadcast_dims - 1 - broadcast_idx

    def _codegen_jit_wrapper_tma(self, ctx: _CodegenContext, kernel_arg: str) -> None:
        code = ctx.code
        kernel_input_params = ctx.kernel_input_params
        output_params = ctx.output_params

        # TMA automatically handles out-of-bounds accesses
        code.writeline("# Use lax.fori_loop with TMA for automatic OOB masking")
        code.writeline("from jax import lax")
        code.writeline("_tile_size = 128  # Warpgroup size")
        code.writeline("_orig_out_shapes = out_shapes")

        code.writeline("_max_numel = 0")
        for param in kernel_input_params:
            code.writeline(f"_max_numel = max(_max_numel, {param}.size)")
        code.writeline("for shape in out_shapes:")
        code.writeline("    _max_numel = max(_max_numel, math.prod(shape))")

        code.writeline("_num_tiles = (_max_numel + _tile_size - 1) // _tile_size")

        gmem_input_params = [f"{p}_gmem" for p in kernel_input_params]
        gmem_output_params = [f"{p}_gmem" for p in output_params]
        smem_input_params = [f"{p}_smem" for p in kernel_input_params]
        smem_output_params = [f"{p}_smem" for p in output_params]

        code.writeline("")
        code.writeline("# Wrapper kernel using lax.fori_loop with direct TMA")

        wrapper_kernel_params = gmem_input_params + gmem_output_params
        all_smem_params = smem_input_params + smem_output_params
        barrier_params = [f"_barrier_{i}" for i in range(len(kernel_input_params))]
        scratch_params = ", ".join(all_smem_params + barrier_params)

        code.writeline(
            f"def _tma_kernel({', '.join(wrapper_kernel_params)}, *, {scratch_params}):"
        )
        with code.indent():
            code.writeline("")
            code.writeline("def _tile_body(_tile_idx, _):")
            with code.indent():
                code.writeline("_tile_start = _tile_idx * _tile_size")
                code.writeline("")

                code.writeline("# TMA load inputs from GMEM to SMEM (OOB auto-masked)")
                for i, (gmem_in, smem_in) in enumerate(
                    zip(gmem_input_params, smem_input_params)
                ):
                    code.writeline(
                        f"plgpu.copy_gmem_to_smem({gmem_in}.at[pl.ds(_tile_start, _tile_size)], {smem_in}, _barrier_{i})"
                    )

                code.writeline("")
                code.writeline("# Wait for TMA loads to complete")
                for i, _ in enumerate(gmem_input_params):
                    code.writeline(f"plgpu.barrier_wait(_barrier_{i})")

                code.writeline("")
                code.writeline("# Compute on SMEM tiles")
                kernel_call_args = smem_input_params + smem_output_params
                kernel_fn = kernel_arg.rstrip(",").strip()
                code.writeline(f"{kernel_fn}({', '.join(kernel_call_args)})")

                code.writeline("")
                code.writeline(
                    "# TMA store outputs from SMEM to GMEM (OOB auto-masked)"
                )
                code.writeline("plgpu.commit_smem()")
                for gmem_out, smem_out in zip(gmem_output_params, smem_output_params):
                    code.writeline(
                        f"plgpu.copy_smem_to_gmem({smem_out}, {gmem_out}.at[pl.ds(_tile_start, _tile_size)])"
                    )
                code.writeline("plgpu.wait_smem_to_gmem(0)")
                code.writeline("")
                code.writeline("return None")

            code.writeline("")
            code.writeline("# Iterate over all tiles")
            code.writeline("lax.fori_loop(0, _num_tiles, _tile_body, None)")

        # Build scratch_shapes dict
        code.writeline("")
        code.writeline(
            "# Build SMEM scratch shapes for inputs, outputs, and TMA barriers"
        )
        code.writeline("_scratch_shapes = {}")
        for i, smem_param in enumerate(smem_input_params):
            orig_param = kernel_input_params[i]
            code.writeline(
                f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), {orig_param}.dtype)"
            )
        for i, smem_param in enumerate(smem_output_params):
            code.writeline(
                f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), out_dtypes[{i}])"
            )
        for barrier_param in barrier_params:
            code.writeline(
                f"_scratch_shapes['{barrier_param}'] = plgpu.Barrier(num_arrivals=1)"
            )

        code.writeline("")
        code.writeline("# Create flattened output specs aligned to tile size")
        code.writeline(
            "_flat_out_specs, _ = pallas_gpu_align_output_specs(out_shapes, out_dtypes, _tile_size)"
        )

        code.writeline("")
        code.writeline("# Call plgpu.kernel with TMA kernel")
        code.writeline("_result = plgpu.kernel(")
        with code.indent():
            code.writeline("_tma_kernel,")
            code.writeline("out_shape=_flat_out_specs,")
            code.writeline("scratch_shapes=_scratch_shapes,")
        code.writeline(")(")
        for param in kernel_input_params:
            code.writeline(f"    {param}.flatten(),")
        code.writeline(")")

        code.writeline("")
        code.writeline("# Reshape results to original shapes")
        code.writeline("return pallas_gpu_unpad_results(_result, _orig_out_shapes)")

    def _codegen_jit_wrapper_legacy_gpu(
        self, ctx: _CodegenContext, kernel_arg: str
    ) -> None:
        code = ctx.code
        kernel_input_params = ctx.kernel_input_params
        input_list = f"[{', '.join(kernel_input_params)}]"

        # Legacy GPU path with explicit padding (use_emit_pipeline=False)
        # Mosaic GPU requires tensor sizes to be multiples of 128.
        # Only apply padding when all tensors have the same size (no broadcasting).
        code.writeline("# Check if all tensors have same size (no broadcasting)")
        code.writeline("_all_sizes = []")
        for param in kernel_input_params:
            code.writeline(f"_all_sizes.append({param}.size)")
        code.writeline("for shape in out_shapes:")
        code.writeline("    _all_sizes.append(math.prod(shape))")
        code.writeline("_unique_sizes = set(_all_sizes)")
        code.writeline(
            "_can_pad = len(_unique_sizes) == 1 and all(s > 1 for s in _unique_sizes)"
        )

        code.writeline("")
        code.writeline("if _can_pad:")
        code.writeline("    # All tensors same size - safe to flatten and pad")
        code.writeline(f"    _padded_inputs = pallas_gpu_pad_inputs({input_list})")
        code.writeline(
            "    _aligned_out_specs, _is_scalar = pallas_gpu_align_output_specs(out_shapes, out_dtypes)"
        )
        code.writeline("    _result = plgpu.kernel(")
        code.writeline("        " + kernel_arg)
        code.writeline("        out_shape=_aligned_out_specs,")
        code.writeline("    )(*_padded_inputs)")
        code.writeline(
            "    return pallas_gpu_unpad_results(_result, out_shapes, _is_scalar)"
        )

        code.writeline("else:")
        code.writeline(
            "    # Different sizes - check if it's a reduction (scalar output)"
        )
        code.writeline("    _out_numel = math.prod(out_shapes[0])")
        code.writeline("    ")
        code.writeline("    if _out_numel <= 1:")
        code.writeline(
            "        # Scalar output (reduction) - pad inputs but keep scalar output"
        )
        code.writeline(f"        _padded_inputs = pallas_gpu_pad_inputs({input_list})")
        code.writeline("        _aligned_out_specs = tuple(")
        code.writeline("            jax.ShapeDtypeStruct(shape, dtype)")
        code.writeline("            for shape, dtype in zip(out_shapes, out_dtypes)")
        code.writeline("        )")
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
        code.writeline("        _broadcasted = [")
        code.writeline(
            f"            jnp.broadcast_to(_inp, _target_shape) for _inp in {input_list}"
        )
        code.writeline("        ]")
        code.writeline("        _padded_inputs = pallas_gpu_pad_inputs(_broadcasted)")
        code.writeline(
            "        _aligned_out_specs, _is_scalar = pallas_gpu_align_output_specs(out_shapes, out_dtypes)"
        )
        code.writeline("        _result = plgpu.kernel(")
        code.writeline("            " + kernel_arg)
        code.writeline("            out_shape=_aligned_out_specs,")
        code.writeline("        )(*_padded_inputs)")
        code.writeline(
            "        return pallas_gpu_unpad_results(_result, out_shapes, _is_scalar)"
        )

    def _codegen_tiled_specs(self, ctx: _CodegenContext) -> None:
        """Generate tiled BlockSpec and grid variables for CPU/TPU.

        Tiles the last 1–2 dimensions of each tensor, respecting TPU
        alignment constraints (last dim multiple of 128, second-to-last
        multiple of 8).  Lower-ndim inputs are right-aligned with the
        reference output shape per numpy broadcast rules.
        """
        code = ctx.code
        transpose_literal = "True" if self.tile_has_transpose else "False"

        skip_n = self.tile_skip_last_n
        code.writeline(
            f"_tile, _grid, _ax2g = pallas_compute_tiling("
            f"out_shapes[0], transpose={transpose_literal}, "
            f"skip_last_n={skip_n}, exact_only=True)"
        )
        code.writeline("_ng = len(_grid)")
        code.writeline("_ref = out_shapes[0]")

        code.writeline("out_specs_pallas = tuple(")
        code.writeline(
            "    pallas_make_block_spec(s, _ref, _tile, _ax2g, _ng, is_output=True)"
        )
        code.writeline("    for s in out_shapes")
        code.writeline(")")

        # Build per-input swap_last_two flags for transposed buffers
        swap_flags = []
        for param in ctx.kernel_input_params:
            # Map kernel param back to graph buffer name
            buf_name = None
            for graph_name, inner_name in self.args.input_buffers.items():
                if inner_name == param:
                    buf_name = graph_name
                    break
            is_swap = buf_name is not None and buf_name in self.transposed_input_buffers
            swap_flags.append(is_swap)

        if any(swap_flags):
            swap_list = ", ".join(str(f) for f in swap_flags)
            code.writeline(f"_swap_flags = [{swap_list}]")
            input_list = ", ".join(ctx.kernel_input_params)
            code.writeline("in_specs_pallas = tuple(")
            code.writeline(
                f"    pallas_make_block_spec(i.shape, _ref, _tile, _ax2g, _ng, swap_last_two=s)"
                f" for i, s in zip([{input_list}], _swap_flags)"
            )
            code.writeline(")")
        else:
            input_list = ", ".join(ctx.kernel_input_params)
            code.writeline("in_specs_pallas = tuple(")
            code.writeline(
                f"    pallas_make_block_spec(i.shape, _ref, _tile, _ax2g, _ng) for i in [{input_list}]"
            )
            code.writeline(")")

    def _codegen_jit_wrapper_cpu_tpu(
        self,
        ctx: _CodegenContext,
        kernel_arg: str,
        alias_pairs: list[tuple[int, int]],
        alias_map_literal: str,
    ) -> None:
        code = ctx.code
        grid_expr = "_grid" if self.tile_cpu_tpu else "(1,)"
        code.writeline("_result = pl.pallas_call(")
        code.writeline("    " + kernel_arg)
        code.writeline("    out_shape=out_shapes_pallas,")
        code.writeline("    out_specs=out_specs_pallas,")
        code.writeline("    in_specs=in_specs_pallas,")
        code.writeline(f"    interpret={ctx.interpret_literal},")
        code.writeline(f"    grid={grid_expr},")
        code.writeline(
            f"    input_output_aliases={{ {alias_map_literal} }},"
            if alias_pairs
            else "    input_output_aliases={},"
        )
        code.writeline(")(")
        if ctx.kernel_input_params:
            code.writeline(f"    {', '.join(ctx.kernel_input_params)},")
        code.writeline(")")
        # Reshape results back to original shapes (restores 0-d from promoted (1,))
        code.writeline("if isinstance(_result, tuple):")
        code.writeline(
            "    _result = tuple(r.reshape(s) for r, s in zip(_result, out_shapes))"
        )
        code.writeline("else:")
        code.writeline("    _result = _result.reshape(out_shapes[0])")
        code.writeline("return _result")

    def _codegen_main_entry(self, ctx: _CodegenContext, jit_wrapper_name: str) -> None:
        if ctx.is_tpu:
            self._codegen_main_entry_tpu(ctx, jit_wrapper_name)
        else:
            self._codegen_main_entry_default(ctx, jit_wrapper_name)

    def _codegen_main_entry_tpu(
        self, ctx: _CodegenContext, jit_wrapper_name: str
    ) -> None:
        code = ctx.code
        code.writeline("")
        main_name = f"{ctx.kernel_name}_main"
        kernel_name_str = ctx.kernel_name
        code.writeline(
            f"def {main_name}({', '.join(ctx.full_kernel_params)}, stream=None):"
        )
        with code.indent():
            code.writeline("jax.config.update('jax_enable_x64', True)")
            code.writeline("jax.clear_caches()")

            # Build JAX placeholders for all inputs
            code.writeline("# Build JAX placeholders for export tracing")
            all_jax_input_names = []
            for alias_name in ctx.alias_params:
                code.writeline(
                    f"{alias_name}_placeholder = jax.ShapeDtypeStruct("
                    f"{alias_name}.shape, torch_dtype_to_jax_runtime({alias_name}.dtype))"
                )
                all_jax_input_names.append(f"{alias_name}_placeholder")
            for ptr in ctx.pointer_tail:
                code.writeline(
                    f"{ptr}_placeholder = jax.ShapeDtypeStruct("
                    f"{ptr}.shape, torch_dtype_to_jax_runtime({ptr}.dtype))"
                )
                all_jax_input_names.append(f"{ptr}_placeholder")

            # Prepare output metadata
            code.writeline(
                "out_shapes = ("
                + ", ".join([f"tuple({name}.shape)" for name in ctx.output_params])
                + ",)"
            )
            dtype_exprs: list[str] = []
            for name in ctx.output_params:
                buf_name = ctx.output_buffer_lookup.get(name)
                if buf_name is not None:
                    dtype = V.graph.get_dtype(buf_name)
                    if dtype is not None:
                        dtype_exprs.append(torch_dtype_to_jax(dtype))
                        continue
                dtype_exprs.append(f"torch_dtype_to_jax_runtime({name}.dtype)")
            code.writeline("out_dtypes = (" + ", ".join(dtype_exprs) + ",)")

            # Export the jit_wrapper
            wrapper_placeholder_args = ["out_shapes", "out_dtypes"]
            wrapper_placeholder_args.extend(ctx.size_var_params)
            wrapper_placeholder_args.extend(all_jax_input_names)
            code.writeline(
                f"exported = jax.export.export("
                f"{jit_wrapper_name}, platforms=['tpu'])"
                f"({', '.join(wrapper_placeholder_args)})"
            )

            # Register and call via tpu_torch_pallas
            # Include all output and input shapes in the key to avoid stale
            # cache hits when the same kernel name is compiled with different
            # input/output ranks (e.g. broadcasting vs non-broadcasting calls).
            shape_key_parts = []
            for p in ctx.output_params:
                shape_key_parts.append(f"'_'.join(str(s) for s in {p}.shape)")
            output_key_expr = (
                " + 'x' + ".join(shape_key_parts) if shape_key_parts else "''"
            )
            input_key_parts = []
            for p in ctx.kernel_input_params:
                input_key_parts.append(f"'_'.join(str(s) for s in {p}.shape)")
            input_key_expr = (
                " + 'x' + ".join(input_key_parts) if input_key_parts else "''"
            )
            code.writeline(
                f"kernel_key = '{kernel_name_str}_out_' + "
                f"{output_key_expr}"
                f" + '_in_' + {input_key_expr}"
            )

            code.writeline(
                f"if not tpu_torch_pallas.lookup_custom_kernel('{kernel_name_str}', kernel_key):"
            )
            with code.indent():
                code.writeline(
                    f"tpu_torch_pallas.register_custom_kernel("
                    f"'{kernel_name_str}', kernel_key, exported.mlir_module_serialized)"
                )

            # Build input tensor list (all non-size-var inputs)
            input_tensor_names = list(ctx.alias_params) + list(ctx.pointer_tail)
            code.writeline(f"input_tensors = [{', '.join(input_tensor_names)}]")

            # Build output shapes list
            code.writeline("output_shape_tensors = [")
            with code.indent():
                for name in ctx.output_params:
                    buf_name = ctx.output_buffer_lookup.get(name)
                    if buf_name is not None:
                        dtype = V.graph.get_dtype(buf_name)
                        if dtype is not None:
                            code.writeline(
                                f"torch.empty({name}.shape, dtype={dtype!r}, device='tpu'),"
                            )
                            continue
                    code.writeline(
                        f"torch.empty({name}.shape, dtype={name}.dtype, device='tpu'),"
                    )
            code.writeline("]")

            # Build input_output_aliases for zero-copy donation
            if ctx.alias_pairs:
                alias_map_str = ", ".join(f"{i}: {o}" for (i, o) in ctx.alias_pairs)
                code.writeline(f"_input_output_aliases = {{ {alias_map_str} }}")
            else:
                code.writeline("_input_output_aliases = {}")

            code.writeline(
                f"tpu_torch_pallas.call_custom_kernel("
                f"input_tensors, output_shape_tensors, "
                f"'{kernel_name_str}', kernel_key, _input_output_aliases)"
            )

    def _codegen_main_entry_default(
        self, ctx: _CodegenContext, jit_wrapper_name: str
    ) -> None:
        code = ctx.code
        code.writeline("")
        main_name = f"{ctx.kernel_name}_main"
        code.writeline(
            f"def {main_name}({', '.join(ctx.full_kernel_params)}, stream=None):"
        )
        with code.indent():
            code.writeline("jax.config.update('jax_enable_x64', True)")
            code.writeline("jax.clear_caches()")
            if ctx.alias_params:
                code.writeline("# Convert Torch -> JAX for donated outputs")
                for alias_name in ctx.alias_params:
                    # On CPU/TPU, alias outputs may be non-contiguous (e.g.
                    # torch.cat slices) and JAX's from_dlpack rejects
                    # non-trivially strided tensors.  Making them contiguous
                    # is safe because CPU/TPU already copies all results back
                    # via copy_output_indices.  On CUDA, the donated-buffer
                    # mechanism requires the original buffer for in-place
                    # mutation, so we cannot make a contiguous copy.
                    self._emit_torch_to_jax(
                        code,
                        alias_name,
                        ctx.is_tpu,
                        contiguous=ctx.interpret_is_cpu,
                    )
            code.writeline("# Convert Torch -> JAX for in-place tensors")
            for ptr in ctx.pointer_tail:
                if ptr.startswith("in_out_ptr"):
                    self._emit_torch_to_jax(code, ptr, ctx.is_tpu, contiguous=False)
            code.writeline("# Convert Torch -> JAX for inputs")
            for ptr in ctx.pointer_tail:
                if ptr.startswith("in_ptr"):
                    self._emit_torch_to_jax(code, ptr, ctx.is_tpu, contiguous=True)

            code.writeline("# Prepare output metadata from PyTorch tensor")
            code.writeline(
                "out_shapes = ("
                + ", ".join([f"tuple({name}.shape)" for name in ctx.output_params])
                + ",)"
            )
            dtype_exprs: list[str] = []
            for name in ctx.output_params:
                buf_name = ctx.output_buffer_lookup.get(name)
                if buf_name is not None:
                    dtype = V.graph.get_dtype(buf_name)
                    if dtype is not None:
                        dtype_exprs.append(torch_dtype_to_jax(dtype))
                        continue
                dtype_exprs.append(f"torch_dtype_to_jax_runtime({name}.dtype)")
            code.writeline("out_dtypes = (" + ", ".join(dtype_exprs) + ",)")
            arg_name_map: dict[str, str] = {}
            for alias_name in ctx.alias_params:
                arg_name_map[alias_name] = f"{alias_name}_jax"
            for ptr in ctx.pointer_tail:
                arg_name_map[ptr] = f"{ptr}_jax"

            wrapper_call_args = ["out_shapes", "out_dtypes"]
            wrapper_call_args.extend(ctx.size_var_params)
            wrapper_call_args.extend(
                arg_name_map[name] for name in ctx.kernel_input_params
            )
            code.writeline(f"res = {jit_wrapper_name}({', '.join(wrapper_call_args)})")
            code.writeline("jax.block_until_ready(res)")
            if ctx.copy_output_indices:
                code.writeline(
                    "result_values = res if isinstance(res, tuple) else (res,)"
                )
                for idx in ctx.copy_output_indices:
                    out_name = ctx.output_params[idx]
                    code.writeline(
                        f"{out_name}.copy_(torch.from_dlpack(result_values[{idx}]))"
                    )

    @staticmethod
    def _emit_torch_to_jax(
        code: IndentedBuffer, var_name: str, is_tpu: bool, *, contiguous: bool
    ) -> None:
        suffix = ".detach().contiguous()" if contiguous else ".detach()"
        code.writeline(f"{var_name}_jax = jax.dlpack.from_dlpack({var_name}{suffix})")

    def call_kernel(self, name: str, node: IRNode | None = None) -> None:  # type: ignore[override]
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
