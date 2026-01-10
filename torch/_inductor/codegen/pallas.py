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
        return f"(1.0 / jnp.sqrt({x}))"

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
        return f"(1.0 / (1.0 + jnp.exp(-{x})))"

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
    - Use masked loads/stores with power-of-2 block sizes to handle non-power-of-2 shapes
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pallas_pexpr  # Use Pallas expression printer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine device type once at initialization
        device = V.graph.get_current_device_or_throw()
        self.is_gpu = device.type == "cuda"
        self.use_masked_ops: bool | None = None
        # Enable warpgroup padding for GPU to handle non-aligned tensor sizes
        # Mosaic GPU requires tensor sizes to be multiples of 128 (WARPGROUP_SIZE)
        self.use_warpgroup_padding = self.is_gpu
        self.tensor_masks = {}  # Map tensor name to mask variable name
        # Track which output param each store uses: list of (out_ptr_name, store_line)
        self.store_with_output: list[tuple[str, str]] = []
        # Track load index expressions for argmax/argmin axis detection
        self.load_index_exprs: dict[str, sympy.Expr] = {}
        # Track outputs that need to be readable (for scatter operations)
        self.outputs_need_read: OrderedSet[str] = OrderedSet()
        # Track if any load in this kernel used transpose
        # Used to avoid double transpose (load + store)
        self.has_transposed_load = False

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

    def _is_transposed_access(self, name: str, index: sympy.Expr) -> bool:
        """Check if buffer access needs transpose.

        Transpose on load is needed when:
        1. Non-square buffers: dimensions are swapped relative to iteration vars
        2. Square buffers: index coefficient pattern indicates transposed access
           (first iteration var has larger coefficient than second)
        """
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return False

        buf_size = buf_obj.get_size()

        # Only handle 2D buffers
        if len(buf_size) != 2:
            return False

        layout = getattr(buf_obj, "get_layout", lambda: None)()
        if layout is None:
            return False

        buf_stride = getattr(layout, "stride", None)
        if buf_stride is None or len(buf_stride) != 2:
            return False

        size0 = self._safe_int(buf_size[0])
        size1 = self._safe_int(buf_size[1])
        if size0 is None or size1 is None or size0 <= 1 or size1 <= 1:
            return False

        # Get buffer strides
        s0 = self._safe_int(buf_stride[0])
        s1 = self._safe_int(buf_stride[1])
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

        inner_coeff = get_coefficient(index, inner_var)
        outer_coeff = get_coefficient(index, outer_var)

        if inner_coeff is not None and outer_coeff is not None:
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

    def _determine_masked_ops_for_kernel(self) -> bool:
        """
        Determine if we should use masked ops for this entire kernel.

        Masked ops with pl.ds(block_size) flatten tensors to 1D, which works when:
        1. We're on GPU (CUDA backend uses Mosaic which requires power-of-2 sizes)
        2. All tensors are already 1D (so flattening doesn't change dimensionality)
        3. All tensors have the same size (so broadcasting works correctly)

        With per-tensor masks, each tensor gets its own mask based on its size.

        This should be called once in codegen_kernel() before generating the kernel body.
        """
        # Mosaic GPU backend doesn't support jnp.arange inside kernels,
        # so we can't use masked ops which require creating mask arrays.
        # CPU doesn't need masked ops either.
        # TODO: Re-enable masked ops when Mosaic supports the required operations
        return False

        # Get all buffer sizes
        # We need ALL buffers - inputs, outputs, and intermediates
        all_buffer_names = OrderedSet()

        # Get input buffers from args
        all_buffer_names.update(self.args.input_buffers.keys())
        # Get output buffers from args
        all_buffer_names.update(self.args.output_buffers.keys())
        # Also get any intermediate buffers from the graph
        all_buffer_names.update(V.graph.name_to_buffer.keys())

        # Get shapes and sizes for all buffers
        # Use try/except to handle special layouts (MultiOutputLayout, NoneLayout, etc.)
        # that don't support get_size()
        buf_info = []
        for buf_name in all_buffer_names:
            try:
                buf = V.graph.get_buffer(buf_name)
                if buf is None:
                    continue
                size = buf.get_size()
                shape = tuple(self._safe_int(s) or s for s in size)
                # Calculate flattened size
                total_size = 1
                for s in size:
                    int_s = self._safe_int(s)
                    total_size *= int_s if int_s is not None else s
                buf_info.append((buf_name, shape, total_size))
            except Exception:
                pass

        # Use masked ops when:
        # 1. We're on GPU (Mosaic requires power-of-2 sizes)
        # 2. All buffers have the same flattened size (for correct broadcasting)
        # 3. Any dimension has non-power-of-2 size
        #
        # For multi-D tensors, we flatten to 1D and use masked loads/stores.
        # The mask handles out-of-bounds elements when padding to power-of-2.
        if buf_info and len(buf_info) > 0:
            # Check if all have the same flattened size
            first_size = buf_info[0][2]
            all_same_size = all(size == first_size for _, _, size in buf_info)
            if not all_same_size:
                return False

            # Check if any dimension is non-power-of-2
            def is_power_of_2(n):
                return n > 0 and (n & (n - 1)) == 0

            has_non_pow2 = False
            for _, shape, _ in buf_info:
                for dim in shape:
                    if isinstance(dim, int) and not is_power_of_2(dim):
                        has_non_pow2 = True
                        break
                if has_non_pow2:
                    break

            # Use masked ops if any dimension is non-power-of-2
            return has_non_pow2

        return False

    def _get_or_create_mask(self, buf_name: str) -> str:
        """Get or create a unique mask variable for a buffer."""
        if buf_name not in self.tensor_masks:
            mask_var = f"mask_{buf_name}"
            self.tensor_masks[buf_name] = mask_var
        return self.tensor_masks[buf_name]

    def _ensure_masked_ops_initialized(self) -> None:
        """Initialize masked ops strategy on first load/store if not yet determined."""
        if self.use_masked_ops is None:
            self.use_masked_ops = self._determine_masked_ops_for_kernel()

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
        use_masked: bool,
    ) -> str:
        """
        Build the load expression based on indexing mode.
        """
        if use_masked:
            # GPU masked load: flatten tensor and apply per-tensor mask
            mask_var = self._get_or_create_mask(name)
            return f"pltriton.load({buf}.at[pl.ds(block_size)], mask={mask_var})"
        elif needs_flatten:
            # Flatten then index for non-contiguous access (gather operation)
            if self.is_gpu:
                # GPU: use pltriton.load with explicit offsets
                return f"pltriton.load({buf}.at[{index_str}])"
            else:
                # CPU: use JAX array indexing
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

    def _build_full_array_store_expr(
        self, out: str, value: CSEVariable, needs_transpose: bool
    ) -> str:
        """
        Build store expression for full array assignment.

        Handles scalar broadcast, shape matching, and optional transpose.
        """
        if needs_transpose:
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.transpose(jnp.asarray({value})))"
            )
        else:
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else (jnp.broadcast_to(jnp.asarray({value}), {out}.shape) "
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
        use_masked: bool,
        mode: Any = None,
    ) -> str:
        """
        Build the store expression based on indexing mode.
        mode can be None (set) or "atomic_add" (accumulate).
        """
        if use_masked:
            # GPU masked store: flatten tensor and apply per-tensor mask
            mask_var = self._get_or_create_mask(name)
            return (
                f"pltriton.store({out}.at[pl.ds(block_size)], {value}, mask={mask_var})"
            )

        if index_str == "...":
            # Full array store with shape matching
            needs_transpose = self._check_store_needs_transpose(name)
            return self._build_full_array_store_expr(out, value, needs_transpose)

        if needs_flatten:
            # Block variable indexing (e.g., im2col) - use flattened scatter
            scatter_op = "add" if mode == "atomic_add" else "set"
            if self.is_gpu:
                return f"pltriton.store({out}.at[{index_str}], jnp.asarray({value}))"
            else:
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
                return self._build_full_array_store_expr(out, value, False)

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

        self._ensure_masked_ops_initialized()

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

        # Determine if masked operations should be used
        use_masked = (
            index_str == "..." and not needs_flatten and self.use_masked_ops is True
        )

        # Build the load expression
        load_expr = self._build_load_expr(
            buf, name, index, index_str, needs_flatten, use_masked
        )

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

        self._ensure_masked_ops_initialized()

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

                # Determine if masked operations should be used
                use_masked = (
                    index_str == "..."
                    and not needs_flatten
                    and self.use_masked_ops is True
                )

                # Build the store expression
                store_expr = self._build_store_expr(
                    out, name, index, value, index_str, needs_flatten, use_masked, mode
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

        # For GPU (Mosaic backend), import plgpu for masked loads/stores
        # Import math for masked ops and symbolic expressions (e.g., math.floor, math.log2)
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
            # For masked ops on GPU, generate per-tensor masks at the start
            if self.use_masked_ops and self.tensor_masks:
                # Create a mapping from buffer name to parameter name
                buf_to_param = {}
                for outer, inner in self.args.input_buffers.items():
                    buf_to_param[outer] = inner if isinstance(inner, str) else outer
                for outer, inner in self.args.output_buffers.items():
                    buf_to_param[outer] = inner if isinstance(inner, str) else outer

                # Generate a mask for each tensor that was accessed
                for buf_name, mask_var in sorted(self.tensor_masks.items()):
                    param_name = buf_to_param.get(buf_name, buf_name)
                    # Find the corresponding parameter in kernel_params
                    matching_param = None
                    for p in kernel_params:
                        # Check if this parameter corresponds to the buffer
                        if param_name == p or buf_name in str(p):
                            matching_param = p
                            break

                    if matching_param:
                        # Calculate flattened size for this tensor
                        kernel_body.writeline(f"# Mask for {buf_name}")
                        kernel_body.writeline(
                            f"{mask_var}_size = {matching_param}.size"
                        )
                        kernel_body.writeline(
                            f"{mask_var} = jnp.arange(block_size) < {mask_var}_size"
                        )

            # Generate iteration variables as jnp.arange arrays
            # These are used by index_expr operations like torch.arange
            # Skip on GPU - jnp.arange is not supported by Pallas Mosaic backend
            # Skip on GPU with masked ops - iteration vars would create non-power-of-2 arrays
            if self.range_tree_nodes and not self.use_masked_ops and not self.is_gpu:
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
        # For GPU with masked ops, add block_size as keyword-only parameter
        kernel_signature = (
            f"def {kernel_name}_kernel({', '.join(full_kernel_params)}"
            + (", *, block_size" if self.use_masked_ops else "")
            + "):"
        )
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

            # For masked ops, calculate block_size aligned to WARPGROUP_SIZE (128)
            # Mosaic GPU runs with 128 threads (1 warpgroup), so data sizes should
            # be at least 128 and aligned to 128 for efficient processing.
            if self.use_masked_ops:
                code.writeline(
                    "# Calculate block_size aligned to warpgroup size (128) for Mosaic GPU"
                )
                code.writeline("# Find maximum flattened size across all tensors")
                code.writeline("max_size = 0")
                # Calculate size for all input tensors
                for param in kernel_input_params:
                    code.writeline(f"max_size = max(max_size, {param}.size)")
                # Also consider output shapes
                code.writeline("for shape in out_shapes:")
                code.writeline(
                    "    tensor_size = shape[0] if len(shape) == 1 else math.prod(shape)"
                )
                code.writeline("    max_size = max(max_size, tensor_size)")
                # Align to WARPGROUP_SIZE (128) and ensure at least 128
                code.writeline(
                    "# Align to warpgroup size (128) for efficient GPU processing"
                )
                code.writeline("block_size = max(128, ((max_size + 127) // 128) * 128)")

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

            # Wrap kernel with functools.partial to pass scalar arguments
            # (size variables and block_size for masked ops)
            partial_args = []
            for sv_param in size_var_params:
                partial_args.append(f"{sv_param}={sv_param}")
            if self.use_masked_ops:
                partial_args.append("block_size=block_size")

            if partial_args:
                kernel_arg = f"functools.partial({kernel_name}_kernel, {', '.join(partial_args)}),"
            else:
                kernel_arg = f"{kernel_name}_kernel,"

            # Use plgpu.kernel for GPU (Mosaic), pl.pallas_call for CPU/TPU
            if self.is_gpu:
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
                        code.writeline(
                            f"{alias_name}_jax = jax.dlpack.from_dlpack({alias_name}.detach())"
                        )
            code.writeline("# Convert Torch -> JAX for in-place tensors")
            for ptr in pointer_tail:
                if ptr.startswith("in_out_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach())"
                        )
            code.writeline("# Convert Torch -> JAX for inputs")
            for ptr in pointer_tail:
                if ptr.startswith("in_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    elif self.use_masked_ops:
                        # For masked ops, flatten inputs to 1D for Mosaic compatibility
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach().contiguous().flatten())"
                        )
                    else:
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach().contiguous())"
                        )

            code.writeline("# Prepare output metadata from PyTorch tensor")
            if self.use_masked_ops:
                # For masked ops, flatten multi-D tensors to 1D for Mosaic compatibility
                code.writeline(
                    "out_shapes = ("
                    + ", ".join(
                        [f"(math.prod({name}.shape),)" for name in output_params]
                    )
                    + ",)"
                )
            else:
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
                    elif self.use_masked_ops:
                        # For masked ops, result is flattened, reshape back to original shape
                        code.writeline(
                            f"{name}.copy_(torch.from_dlpack(result_values[{idx}]).reshape({name}.shape))"
                        )
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
