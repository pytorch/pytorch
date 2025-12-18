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
from ..runtime.runtime_utils import torch_dtype_to_jax
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .block_analysis import BlockPatternMatcher
from .common import BackendFeature, CSEVariable, IndentedBuffer, OpOverrides
from .simd import pexpr, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ir import IRNode
    from ..ops_handler import ReductionType
    from ..scheduler import BaseSchedulerNode


# Main function suffix used in generated Pallas code
MAIN_SUFFIX = "main"

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
        """Fused multiply-add: a * b + c"""
        return f"jnp.fma({a}, {b}, {c})"

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

    For GPU (Triton backend):
    - Use masked loads/stores with power-of-2 block sizes to handle non-power-of-2 shapes
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pexpr  # Use Python expression printer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine device type once at initialization
        device = V.graph.get_current_device_or_throw()
        self.is_gpu = device.type == "cuda"
        self.use_masked_ops: bool | None = None
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

        # Check for ModularIndexing - this is NOT contiguous access
        # ModularIndexing is used for roll/wrap-around operations
        if index.has(ModularIndexing):
            # Generate actual index expression - iteration variables are already
            # defined as jnp.arange arrays, so we just convert to JAX code
            return self.kexpr(index)

        # Simplify the index
        index = V.graph.sizevars.simplify(index)
        free_symbols = index.free_symbols

        # Get iteration variables from range_tree_nodes
        iter_vars = OrderedSet(self.range_tree_nodes.keys())

        # Find which iteration variable(s) are used
        used_vars = free_symbols & iter_vars

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
                # Extract the constant offset (terms not involving var)
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)

                # Generate JAX slice notation
                if stride == 1 and offset == 0:
                    # Contiguous access
                    return "..."
                elif offset == 0:
                    # Pure stride: ::stride
                    stride_str = self.kexpr(stride)
                    return f"::{stride_str}"
                else:
                    # Offset + stride: offset::stride
                    offset_str = self.kexpr(offset)
                    stride_str = self.kexpr(stride)
                    return f"{offset_str}::{stride_str}"
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
        # Get iteration variables
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        free_symbols = index.free_symbols

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

    def _has_iteration_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains iteration variables (x0, x1, etc.)."""
        free_symbols = index.free_symbols
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        return bool(free_symbols & iter_vars)

    def _has_indirect_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains indirect variables (tmp0, tmp1, etc.)."""
        free_symbols = index.free_symbols
        for sym in free_symbols:
            if str(sym).startswith("tmp"):
                return True
        return False

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
            try:
                length = int(entry.length) if hasattr(entry.length, "__int__") else None
                if length is not None:
                    broadcast_vars.append(length)
            except (TypeError, ValueError):
                pass

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
        try:
            buf_obj = V.graph.get_buffer(name)
            buf_size = buf_obj.get_size()
            buf_stride = buf_obj.get_layout().stride

            # Only handle 2D buffers
            if len(buf_size) != 2:
                return False

            size0 = int(buf_size[0]) if hasattr(buf_size[0], "__int__") else None
            size1 = int(buf_size[1]) if hasattr(buf_size[1], "__int__") else None
            if size0 is None or size1 is None or size0 <= 1 or size1 <= 1:
                return False

            # Check buffer stride - if column-major, don't transpose on load
            s0 = int(buf_stride[0]) if hasattr(buf_stride[0], "__int__") else None
            s1 = int(buf_stride[1]) if hasattr(buf_stride[1], "__int__") else None
            if s0 is not None and s1 is not None and s0 < s1:
                return False

            # Get iteration variable info
            var_items = list(self.range_tree_nodes.items())
            if len(var_items) < 2:
                return False

            # Skip for reduction variables
            has_reduction_var = any(
                entry.prefix.startswith("r") for _, entry in var_items
            )
            if has_reduction_var:
                return False

            iter_sizes = []
            for var, entry in var_items:
                length = entry.length
                if hasattr(length, "__int__"):
                    iter_sizes.append(int(length))

            if len(iter_sizes) < 2:
                return False

            # For 2D: iter_sizes[0] is inner, iter_sizes[1] is outer
            expected_rows = iter_sizes[1]
            expected_cols = iter_sizes[0]

            # For non-square buffers: if shape is (cols, rows) instead of (rows, cols)
            if size0 != size1:
                if size0 == expected_cols and size1 == expected_rows:
                    return True
                return False

            # For square buffers: only transpose if output is also column-major
            # This distinguishes actual transpose operations from broadcasting
            # Check by looking at the output buffer name (should end with the store target)
            # We can't directly check output_buffers here, so check the store_mode
            # on the kernel - if it's column-major output, it's transpose

            # Get output buffer info from kernel args (may not be populated yet)
            # Fall back to checking if we have column-major stores expected
            output_is_column_major = False
            try:
                # Check all registered output buffers
                for out_name in getattr(self.args, "output_buffers", {}).values():
                    out_buf = V.graph.get_buffer(out_name)
                    out_stride = out_buf.get_layout().stride
                    if len(out_stride) >= 2:
                        out_s0 = (
                            int(out_stride[0])
                            if hasattr(out_stride[0], "__int__")
                            else None
                        )
                        out_s1 = (
                            int(out_stride[1])
                            if hasattr(out_stride[1], "__int__")
                            else None
                        )
                        if (
                            out_s0 is not None
                            and out_s1 is not None
                            and out_s0 < out_s1
                        ):
                            output_is_column_major = True
                            break
            except Exception:
                pass

            # Only use coefficient analysis if output is column-major
            if not output_is_column_major:
                return False

            first_var = var_items[0][0]
            second_var = var_items[1][0]

            index = V.graph.sizevars.simplify(index)

            def get_coefficient(var):
                if index == var:
                    return 1
                if index.is_Add:
                    for term in index.args:
                        coeff = get_coefficient_from_term(term, var)
                        if coeff is not None:
                            return coeff
                return get_coefficient_from_term(index, var)

            def get_coefficient_from_term(term, var):
                if term == var:
                    return 1
                if term.is_Mul:
                    coeff = 1
                    has_var = False
                    for factor in term.args:
                        if factor == var:
                            has_var = True
                        elif factor.is_number:
                            coeff *= int(factor)
                    if has_var:
                        return coeff
                return None

            coeff_first = get_coefficient(first_var)
            coeff_second = get_coefficient(second_var)

            if coeff_first is not None and coeff_second is not None:
                # Transposed access: first var (inner dim) has larger coefficient
                return coeff_first > coeff_second

            return False

        except Exception:
            return False

    def _get_index_expr(self, index: sympy.Expr) -> tuple[str, bool]:
        """
        Get the index expression string and whether it needs flattening.

        Returns:
            Tuple of (index_str, needs_flatten) where needs_flatten indicates
            if the buffer should be flattened before indexing (for mixed indexing).
        """
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
        1. We're on GPU (CUDA backend uses Triton which requires power-of-2 sizes)
        2. All tensors are already 1D (so flattening doesn't change dimensionality)
        3. All tensors have the same size (so broadcasting works correctly)

        With per-tensor masks, each tensor gets its own mask based on its size.

        This should be called once in codegen_kernel() before generating the kernel body.
        """
        if not self.is_gpu:
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
        buf_info = []
        for buf_name in all_buffer_names:
            try:
                buf = V.graph.get_buffer(buf_name)
                size = buf.get_size()
                shape = tuple(int(s) if hasattr(s, "__int__") else s for s in size)
                # Calculate flattened size
                total_size = 1
                for s in size:
                    if hasattr(s, "__int__"):
                        total_size *= int(s)
                    else:
                        total_size *= s
                buf_info.append((buf_name, shape, total_size))
            except Exception:
                pass

        # Only use masked ops if:
        # 1. All buffers are 1D (single-element shape tuples)
        # 2. All buffers have the same size
        # This ensures that pl.ds(block_size) flattening works correctly
        # and masks can be properly applied without broadcasting issues.
        if buf_info and len(buf_info) > 0:
            # Check if all are 1D
            all_1d = all(len(shape) == 1 for _, shape, _ in buf_info)
            if not all_1d:
                return False

            # Check if all have the same size
            first_size = buf_info[0][2]
            all_same_size = all(size == first_size for _, _, size in buf_info)
            return all_same_size

        return False

    def _get_or_create_mask(self, buf_name: str) -> str:
        """Get or create a unique mask variable for a buffer."""
        if buf_name not in self.tensor_masks:
            mask_var = f"mask_{buf_name}"
            self.tensor_masks[buf_name] = mask_var
        return self.tensor_masks[buf_name]

    @typing_extensions.override
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        # Track the load index expression for argmax/argmin axis detection
        self.load_index_exprs[name] = index

        # Determine masked ops strategy on first load/store if not yet determined
        if self.use_masked_ops is None:
            self.use_masked_ops = self._determine_masked_ops_for_kernel()

        index_str, needs_flatten = self._get_index_expr(index)

        # Check for buffer size mismatch when using full array access
        # This happens with pooling operations where input/output have different sizes
        # In this case, we need to use strided indexing even though _get_index_str
        # returned "..." (because the index expression is strided)
        if index_str == "..." and not needs_flatten:
            try:
                buf_obj = V.graph.get_buffer(name)
                buf_size = buf_obj.get_size()
                buf_numel = 1
                for s in buf_size:
                    buf_numel *= int(s) if hasattr(s, "__int__") else s

                # Get iteration variables used in the index expression
                iter_vars = OrderedSet(self.range_tree_nodes.keys())
                used_vars = index.free_symbols & iter_vars

                # Compute the expected output size from iteration variables USED in the index
                # Only multiply ranges of variables that appear in the index expression
                used_range_lengths = []
                for var in used_vars:
                    if var in self.range_tree_nodes:
                        entry = self.range_tree_nodes[var]
                        try:
                            length_val = (
                                int(entry.length)
                                if hasattr(entry.length, "__int__")
                                else None
                            )
                        except (TypeError, ValueError):
                            length_val = None
                        if length_val is not None:
                            used_range_lengths.append(length_val)

                # Multiply ranges of used variables to get expected output size
                output_numel = 1
                for l in used_range_lengths:
                    output_numel *= l

                # Check if index has gather-pattern strides (im2col-like)
                # For expression like 4*x3 + 224*x4 + y0 + 56*y1 + 3136*y2:
                # - If buffer is contiguous and index coefficients don't match buffer strides,
                #   this is a gather pattern that needs explicit indexing
                # - If index coefficients match buffer strides (like 1 + 10*x for 10xN buffer),
                #   it's standard strided access and [...] is correct after .contiguous()
                has_non_unit_strides = False
                try:
                    buf_stride = buf_obj.get_layout().stride
                    # Check if original buffer is contiguous (stride matches size)
                    is_originally_contiguous = True
                    expected_strides = [1]  # 1D buffers have stride 1
                    if len(buf_size) > 1:
                        expected_stride = 1
                        expected_strides = []  # Reset for multi-dim
                        for i in range(len(buf_size) - 1, -1, -1):
                            expected_strides.insert(0, expected_stride)
                            actual_stride = (
                                int(buf_stride[i])
                                if hasattr(buf_stride[i], "__int__")
                                else None
                            )
                            if (
                                actual_stride is None
                                or actual_stride != expected_stride
                            ):
                                is_originally_contiguous = False
                                break
                            dim_size = (
                                int(buf_size[i])
                                if hasattr(buf_size[i], "__int__")
                                else None
                            )
                            if dim_size is None:
                                is_originally_contiguous = False
                                break
                            expected_stride *= dim_size

                    if is_originally_contiguous:
                        # Buffer is contiguous - check if access coefficients match buffer strides
                        # Coefficients should be a subset of expected strides for normal access
                        coefficients = OrderedSet()
                        for var in used_vars:
                            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(
                                index, var
                            )
                            stride = BlockPatternMatcher.match_affine_block_expr(
                                var_expr, var
                            )
                            if stride is None:
                                stride = 1  # Variable without explicit coefficient has stride 1
                            coefficients.add(
                                int(stride) if hasattr(stride, "__int__") else stride
                            )

                        # If any coefficient is not in expected strides, it's a gather
                        expected_stride_set = OrderedSet(expected_strides)
                        for coef in coefficients:
                            if coef not in expected_stride_set:
                                has_non_unit_strides = True
                                break
                except Exception:
                    pass

                # Use strided indexing if:
                # 1. Buffer size differs from expected output size AND
                # 2. There are iteration variables in the index (meaning we're accessing a subset)
                # OR
                # 3. Not all iteration variables are used (im2col-like patterns)
                #    This handles cases like im2col where load uses 5 vars but store uses 2
                #    The element counts may match but broadcast shapes won't align
                # OR
                # 4. Buffer is contiguous but index has non-unit strides (gather pattern)
                # Note: inputs are made contiguous before passing to JAX, so we don't need
                # to worry about transposed tensors here - the index expression directly
                # corresponds to which elements to load from the contiguous buffer.
                # Only consider "not all vars used" for truly multi-dim buffers.
                # For 1D buffers or 2D broadcast buffers (with a dimension of size 1),
                # using a subset of iteration vars is normal.
                # Also check that the buffer has more dimensions than used vars,
                # which indicates a gather pattern rather than simple broadcast.
                buf_effective_dims = sum(
                    1
                    for s in buf_size
                    if (int(s) if hasattr(s, "__int__") else None) != 1
                )
                # For im2col patterns: the index uses MORE vars than buffer has dims
                # This means multiple iteration vars map to fewer buffer dimensions
                # which is a gather pattern. For normal access (including broadcast),
                # used_vars <= buf_dims.
                not_all_vars_used = (
                    len(used_vars) < len(iter_vars)
                    and len(used_vars) > 0
                    and buf_effective_dims > 1
                    and len(used_vars) > len(buf_size)  # More vars than dims = gather
                )
                if (
                    output_numel > 0
                    and (
                        buf_numel != output_numel
                        or not_all_vars_used
                        or has_non_unit_strides
                    )
                    and len(used_vars) > 0
                ):
                    index_str = self._generate_strided_index(index)
                    needs_flatten = True
            except (TypeError, ValueError, AttributeError):
                pass

        # Build load expression using string concatenation
        use_masked = index_str == "..." and not needs_flatten and self.use_masked_ops

        # Check if we need flattened access for constant indices on multi-dim buffers
        # This is needed for point scatter where a constant index should return a scalar
        if not needs_flatten and index_str != "...":
            try:
                buf_obj = V.graph.get_buffer(name)
                buf_size = buf_obj.get_size()
                # If buffer is 0-dimensional (scalar), use [...] to access it
                # JAX/Pallas doesn't support indexing scalars with [0]
                if len(buf_size) == 0:
                    index_str = "..."
                # If buffer is multi-dimensional and index is a constant/scalar expression,
                # use flattened access to get a single element
                elif len(buf_size) > 1:
                    has_iter_vars = self._has_iteration_vars(index)
                    if not has_iter_vars:
                        needs_flatten = True
                    elif "::" in index_str:
                        # For multi-dim buffers with strided slice patterns (like "::2"),
                        # the slice applies to the first dimension only, which doesn't
                        # match the semantics of the index expression (which operates on
                        # the flattened buffer). Use flattened indexing instead.
                        index_str = self._generate_strided_index(index)
                        needs_flatten = True
            except Exception:
                pass

        if use_masked:
            # GPU masked load: flatten tensor and apply per-tensor mask
            mask_var = self._get_or_create_mask(name)
            load_expr = f"pltriton.load({buf}.at[pl.ds(block_size)], mask={mask_var})"
        elif needs_flatten:
            # Flatten then index for non-contiguous access (gather operation)
            load_expr = f"{buf}[...].flatten()[{index_str}]"
        else:
            # Direct indexing for contiguous access
            load_expr = f"{buf}[{index_str}]"
            # Check for transposed access using index expression coefficients
            # For a 2D buffer with strides [S, 1], normal access is S*outer + inner
            # Transposed access is outer + S*inner (coefficients are swapped)
            if index_str == "...":
                try:
                    needs_transpose = self._is_transposed_access(name, index)
                    if needs_transpose:
                        load_expr = f"jnp.transpose({load_expr})"
                        self.has_transposed_load = True
                except Exception:
                    pass
            # Squeeze (N,1) intermediate buffers when kernel has 1D graph inputs
            # to avoid wrong broadcasting: (N,) op (N,1) -> (N,N) instead of (N,)
            # Check on demand since input_buffers may not be fully populated at __init__ time
            is_intermediate_buffer = name.startswith("buf")
            if is_intermediate_buffer:
                # Check if any input buffer is a 1D graph input
                has_1d_input = False
                for buf_name in self.args.input_buffers:
                    if not buf_name.startswith("buf"):
                        try:
                            buf_obj = V.graph.get_buffer(buf_name)
                            buf_size = buf_obj.get_size()
                            if len(buf_size) == 1:
                                has_1d_input = True
                                break
                        except Exception:
                            pass
                if has_1d_input:
                    try:
                        buf_obj = V.graph.get_buffer(name)
                        buf_size = buf_obj.get_size()
                        if len(buf_size) == 2 and buf_size[-1] == 1:
                            load_expr = f"jnp.squeeze({load_expr}, axis=-1)"
                    except Exception:
                        pass

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
        """
        # Get iteration variables
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        free_symbols = index.free_symbols
        used_iter_vars_set = free_symbols & iter_vars

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
                return 0

        used_iter_vars = sorted(used_iter_vars_set, key=get_coefficient, reverse=True)
        iter_coeffs = [get_coefficient(var) for var in used_iter_vars]

        index_str = self.kexpr(index)
        indirect_var_syms = [s for s in free_symbols if str(s).startswith("tmp")]
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
            is_reduction_var = var in self.range_tree_nodes and self.range_tree_nodes[
                var
            ].prefix.startswith("r")

            if is_reduction_var:
                # Reduction var: simple element-wise gather
                if var in self.range_tree_nodes:
                    range_entry = self.range_tree_nodes[var]
                    range_size = range_entry.length
                    arange_expr = f"jnp.arange({self.kexpr(range_size)})"
                    index_str = index_str.replace(var_name, arange_expr)
                return index_str
            # For pointwise vars, fall through to the complex reshape code

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
                var_coeff = get_coefficient(var)

                arange_expr = f"jnp.arange({self.kexpr(range_size)})"

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

        # Determine masked ops strategy on first load/store if not yet determined
        if self.use_masked_ops is None:
            self.use_masked_ops = self._determine_masked_ops_for_kernel()

        # Check if this is a scalar output (reduction to scalar)
        # Only shape () is a true scalar, not (1,) which is a 1-element tensor
        try:
            buf = V.graph.get_buffer(name)
            output_shape = buf.get_size()
            is_scalar = len(output_shape) == 0
        except Exception:
            output_shape = ()
            is_scalar = False

        if is_scalar:
            # For scalar outputs, use [...] to assign the entire scalar
            store_expr = f"{out}[...] = {value}"
        else:
            # Check for scatter pattern (indirect indexing for stores)
            scatter_info = self._detect_scatter_pattern(index, name)

            if scatter_info is not None:
                is_point_scatter = scatter_info.get("is_point_scatter", False)

                # Mark this output parameter as needing to be readable (for aliasing)
                self.outputs_need_read.add(out)
                alias_param = f"{out}_alias"

                # Use .add() for atomic_add mode (accumulate=True), .set() otherwise
                scatter_op = "add" if mode == "atomic_add" else "set"

                if is_point_scatter:
                    # Single-element scatter: out[fixed_dims..., indirect, fixed_dims...] = scalar
                    indirect_var = scatter_info["indirect_var"]
                    indirect_dim = scatter_info["indirect_dim"]
                    output_shape = scatter_info["output_shape"]

                    # Build index tuple with 0s for other dimensions, indirect_var for scatter dim
                    # For a (2, 3) array with scatter at dim=1: out.at[0, indirect_var].set(val)
                    index_parts = []
                    for dim in range(len(output_shape)):
                        if dim == indirect_dim:
                            index_parts.append(indirect_var)
                        else:
                            index_parts.append("0")

                    index_tuple = ", ".join(index_parts)
                    store_expr = f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"
                else:
                    # Scatter with iteration variables
                    indirect_var = scatter_info["indirect_var"]
                    dims_before = scatter_info["dims_before"]
                    dims_after = scatter_info["dims_after"]

                    # Determine if this is element-wise or slice-based scatter:
                    # - Element-wise: each iter var corresponds to one output dimension
                    #   e.g., scatter(x, 0, ind, src) with x:(196,992), ind:(1,992), src:(1,992)
                    #   Here x0 with range 992 matches output dim 1 exactly
                    # - Slice-based: iter vars together cover multiple output dimensions
                    #   e.g., index_put(a, [b], c) with a:(800,256,7,7), b:(601,), c:(601,256,7,7)
                    #   Here x0 with range 12544=256*7*7 covers dims 1,2,3 together
                    #
                    # Heuristic: if # iter vars in store == # remaining dims, it's element-wise
                    # BUT: if there are more iter vars in the kernel than in the store index,
                    # then some iter vars are embedded in the indirect var, requiring slice scatter
                    try:
                        buf = V.graph.get_buffer(name)
                        output_ndim = len(buf.get_size())
                    except Exception:
                        output_ndim = 0

                    num_iter_vars_in_store = len(dims_before) + len(dims_after)
                    # Total iteration variables in the kernel
                    total_kernel_iter_vars = len(self.range_tree_nodes)
                    # indirect takes 1 dim, iter vars should cover the rest
                    remaining_dims = output_ndim - 1  # dims other than indirect

                    # Element-wise scatter requires:
                    # 1. num iter vars in store == remaining dims
                    # 2. All kernel iter vars appear in store (none embedded in indirect)
                    is_element_wise = (
                        num_iter_vars_in_store == remaining_dims
                        and num_iter_vars_in_store == total_kernel_iter_vars
                    )

                    if is_element_wise:
                        # Element-wise scatter: use iteration variable names
                        # For 2D output with 1 iter var: no reshaping needed (both 1D)
                        # For 3D+ with multiple iter vars: reshape indirect var for broadcasting
                        # e.g., for 3D output with indirect at dim 1 and iter vars at dims 0,2:
                        #   iter vars are reshaped to (n,1,1) and (1,1,m)
                        #   indirect_var shape (k,) needs to become (1, k, 1)
                        index_parts = []
                        for var_name, size in dims_before:
                            index_parts.append(var_name)

                        # Reshape indirect var only if needed for broadcasting with
                        # multi-dimensional iter vars (i.e., more than 1 iter var)
                        n_leading = len(dims_before)
                        n_trailing = len(dims_after)
                        if n_leading > 0 and n_trailing > 0:
                            # Middle dimension: needs reshaping for both before and after
                            leading_ones = "None, " * n_leading
                            trailing_nones = ", None" * n_trailing
                            indirect_reshaped = (
                                f"{indirect_var}[{leading_ones}...{trailing_nones}]"
                            )
                        else:
                            # First or last dimension: no reshaping needed
                            indirect_reshaped = indirect_var
                        index_parts.append(indirect_reshaped)

                        for var_name, size in dims_after:
                            index_parts.append(var_name)
                    else:
                        # Slice-based scatter: use : for iteration dimensions
                        index_parts = []
                        for var_name, size in dims_before:
                            index_parts.append(":")
                        index_parts.append(indirect_var)
                        for var_name, size in dims_after:
                            index_parts.append(":")

                    index_tuple = ", ".join(index_parts)
                    store_expr = f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"
            else:
                index_str, needs_flatten = self._get_index_expr(index)

                # Check for im2col-like patterns where store uses block variables
                # but load doesn't. For cat/expand:
                # - Load uses x0, store uses x0+128*x1 → x2
                # - Both load and store prepared indices share the same block vars
                # - OR: Load has no vars (broadcasts), store has block vars
                # For im2col:
                # - Load uses 4*x3+224*x4+... (NOT compressed, coefficients irregular)
                # - Store uses x3+14*x4+... → x6+196*y5 (compressed, coefficients form products)
                # - Store prepared uses block vars, load prepared doesn't
                if index_str == "..." and not needs_flatten:
                    try:
                        prepared_index = self.prepare_indexing(index)
                        iter_vars = OrderedSet(self.range_tree_nodes.keys())
                        store_orig_vars = index.free_symbols & iter_vars
                        store_prep_vars = (
                            prepared_index.free_symbols
                            if hasattr(prepared_index, "free_symbols")
                            else OrderedSet()
                        ) & iter_vars
                        new_vars = store_prep_vars - store_orig_vars
                        # Only trigger if store introduces new block vars
                        if new_vars and len(store_orig_vars) > 1:
                            # Check if loads are compatible with broadcast or cat pattern
                            # cat: load orig vars ⊂ store orig vars (load uses subset)
                            # im2col: load orig vars = store orig vars but load doesn't compress
                            has_im2col_pattern = False
                            for load_index in self.load_index_exprs.values():
                                load_orig_vars = load_index.free_symbols & iter_vars
                                if load_orig_vars:
                                    # Load has iteration variables
                                    # cat: load vars are strict subset of store vars
                                    # im2col: load vars equal store vars but load doesn't compress
                                    if load_orig_vars == store_orig_vars:
                                        # Same vars - check if load gets compressed too
                                        prep_load = self.prepare_indexing(load_index)
                                        load_prep_vars = (
                                            prep_load.free_symbols
                                            if hasattr(prep_load, "free_symbols")
                                            else OrderedSet()
                                        ) & iter_vars
                                        # If store compresses but load doesn't, it's im2col
                                        if (
                                            load_orig_vars == load_prep_vars
                                            and store_prep_vars != store_orig_vars
                                        ):
                                            has_im2col_pattern = True
                                            break
                            if has_im2col_pattern:
                                index_str = self._generate_strided_index(prepared_index)
                                needs_flatten = True
                    except Exception:
                        pass

                # Build store expression using string concatenation
                use_masked = (
                    index_str == "..." and not needs_flatten and self.use_masked_ops
                )

                if use_masked:
                    # GPU masked store: flatten tensor and apply per-tensor mask
                    mask_var = self._get_or_create_mask(name)
                    store_expr = f"pltriton.store({out}.at[pl.ds(block_size)], {value}, mask={mask_var})"
                elif index_str == "...":
                    # When storing the full array, we need to match the output shape.
                    # This handles:
                    # - Mixed indexing producing flat results needing reshape
                    # - Squeeze operations where value has more dims than output
                    # - Scalar values that need to be broadcast to the output shape
                    # - Expand/broadcast operations (e.g., cat((x, x), 0) -> expand)
                    # - Transpose for column-major outputs when computation is row-major
                    # - If shapes already match, operations are no-ops.
                    # Use jnp.full for scalars, broadcast_to for broadcast-compatible shapes,
                    # and reshape for same-size different-shape arrays.

                    # Check if output needs transpose:
                    # - Output has column-major stride (s0 < s1)
                    # - But input(s) have row-major stride
                    # - This indicates an explicit transpose operation in the IR
                    # If inputs were also column-major, no transpose needed (.copy handles it)
                    # Also skip if we already transposed on load (avoid double transpose)
                    needs_transpose = False
                    if not self.has_transposed_load:
                        try:
                            buf = V.graph.get_buffer(name)
                            buf_stride = buf.get_layout().stride
                            buf_size = buf.get_size()
                            if len(buf_stride) == 2 and len(buf_size) == 2:
                                size0 = (
                                    int(buf_size[0])
                                    if hasattr(buf_size[0], "__int__")
                                    else None
                                )
                                size1 = (
                                    int(buf_size[1])
                                    if hasattr(buf_size[1], "__int__")
                                    else None
                                )
                                s0 = (
                                    int(buf_stride[0])
                                    if hasattr(buf_stride[0], "__int__")
                                    else None
                                )
                                s1 = (
                                    int(buf_stride[1])
                                    if hasattr(buf_stride[1], "__int__")
                                    else None
                                )
                                # Output is column-major
                                if (
                                    s0 is not None
                                    and s1 is not None
                                    and s0 < s1
                                    and size0 is not None
                                    and size1 is not None
                                    and size0 > 1
                                    and size1 > 1
                                ):
                                    # Check if any input is column-major
                                    any_input_column_major = False
                                    for inp_name in self.args.input_buffers:
                                        try:
                                            inp_buf = V.graph.get_buffer(inp_name)
                                            inp_stride = inp_buf.get_layout().stride
                                            if len(inp_stride) == 2:
                                                inp_s0 = (
                                                    int(inp_stride[0])
                                                    if hasattr(inp_stride[0], "__int__")
                                                    else None
                                                )
                                                inp_s1 = (
                                                    int(inp_stride[1])
                                                    if hasattr(inp_stride[1], "__int__")
                                                    else None
                                                )
                                                if (
                                                    inp_s0 is not None
                                                    and inp_s1 is not None
                                                    and inp_s0 < inp_s1
                                                ):
                                                    any_input_column_major = True
                                                    break
                                        except Exception:
                                            pass
                                    # Transpose only if output is column-major but no inputs are
                                    if not any_input_column_major:
                                        needs_transpose = True
                        except Exception:
                            pass

                    if needs_transpose:
                        store_expr = (
                            f"{out}[...] = ("
                            f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                            f"else jnp.transpose(jnp.asarray({value})))"
                        )
                    else:
                        store_expr = (
                            f"{out}[...] = ("
                            f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                            f"else (jnp.broadcast_to(jnp.asarray({value}), {out}.shape) "
                            f"if jnp.asarray({value}).size != {out}.size "
                            f"else jnp.asarray({value}).reshape({out}.shape)))"
                        )
                elif needs_flatten:
                    # Block variable indexing (e.g., im2col) - use flattened scatter
                    # The index_str contains an expression like "x6 + 196*y5" which computes
                    # flat indices. Both the index and value need to be flattened.
                    store_expr = (
                        f"{out}[...] = {out}[...].flatten().at[({index_str}).flatten()].set("
                        f"jnp.asarray({value}).flatten()).reshape({out}.shape)"
                    )
                else:
                    # Direct indexed assignment
                    # Check if we need special handling for constant indices on multi-dim outputs
                    # e.g., storing a scalar to a (1,1,1) output with index 0
                    has_indirect = self._has_indirect_vars(index)
                    try:
                        buf = V.graph.get_buffer(name)
                        buf_size = buf.get_size()
                        if len(buf_size) > 1 and not self._has_iteration_vars(index):
                            # Multi-dim output with constant index - use [...] for full assignment
                            # This handles cases like out_ptr0[0] where output is (1,1,1)
                            store_expr = (
                                f"{out}[...] = ("
                                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                                f"else (jnp.broadcast_to(jnp.asarray({value}), {out}.shape) "
                                f"if jnp.asarray({value}).size != {out}.size "
                                f"else jnp.asarray({value}).reshape({out}.shape)))"
                            )
                        elif has_indirect:
                            # Indirect indexed store (scatter): arr[indices] = value
                            # When value is scalar but indices is an array, JAX requires
                            # the value to match the indexed result shape. Broadcast scalar.
                            store_expr = (
                                f"{out}[{index_str}] = ("
                                f"jnp.full({index_str}.shape, {value}) "
                                f"if jnp.asarray({value}).ndim == 0 else {value})"
                            )
                        else:
                            store_expr = f"{out}[{index_str}] = {value}"
                    except Exception:
                        store_expr = f"{out}[{index_str}] = {value}"

        self.stores.writeline(store_expr)
        # Track which output param this store uses for filtering in codegen_kernel
        self.store_with_output.append((out, store_expr))

    def _detect_scatter_pattern(
        self, index: sympy.Expr, output_name: str = ""
    ) -> Optional[dict]:
        """
        Detect if the index expression represents a scatter operation.

        Scatter patterns occur when:
        1. There's an indirect variable (tmp*) in the index
        2. Optionally, iteration variables cover other dimensions

        Returns:
            dict with keys:
                - 'indirect_var': name of indirect variable
                - 'indirect_dim': which dimension it indexes (0-based from output shape)
                - 'dims_before': list of (var_name, size) for dims before indirect
                - 'dims_after': list of (var_name, size) for dims after indirect
                - 'is_point_scatter': True if single-element scatter (no iter vars)
                - 'output_shape': shape of output buffer (for point scatter)
            or None if not a scatter pattern
        """
        has_indirect = self._has_indirect_vars(index)
        has_iter_vars = self._has_iteration_vars(index)

        if not has_indirect:
            return None

        # Get iteration and indirect variables
        iter_vars = OrderedSet(self.range_tree_nodes.keys())
        free_symbols = index.free_symbols
        used_iter_vars = free_symbols & iter_vars
        indirect_var_syms = [s for s in free_symbols if str(s).startswith("tmp")]

        if len(indirect_var_syms) != 1:
            # Only handle single indirect variable for now
            return None

        indirect_sym = indirect_var_syms[0]
        indirect_var = str(indirect_sym)

        # Get coefficient of each variable
        def get_coefficient(var):
            coeff = index.coeff(var)
            if coeff == 0:
                coeff = sympy.diff(index, var)
            try:
                return int(coeff)
            except (TypeError, ValueError):
                return 0

        indirect_coeff = get_coefficient(indirect_sym)
        if indirect_coeff == 0:
            return None

        # Handle point scatter (no iteration variables)
        # This is single-element scatter where indirect var indexes one dimension
        if not has_iter_vars:
            # Try to get output shape to determine which dimension indirect indexes
            output_shape = None
            if output_name:
                try:
                    buf = V.graph.get_buffer(output_name)
                    output_shape = [int(s) for s in buf.get_size()]
                except Exception:
                    pass

            if output_shape is None or len(output_shape) < 2:
                return None

            # Determine which dimension the indirect var indexes based on coefficient
            # coefficient = product of sizes of all following dimensions
            # For a (2, 3) array: dim 0 has coeff 3, dim 1 has coeff 1
            cumulative_size = 1
            indirect_dim = len(output_shape) - 1  # default to last dim
            for dim in range(len(output_shape) - 1, -1, -1):
                if indirect_coeff == cumulative_size:
                    indirect_dim = dim
                    break
                cumulative_size *= output_shape[dim]

            return {
                "indirect_var": indirect_var,
                "indirect_dim": indirect_dim,
                "dims_before": [],
                "dims_after": [],
                "is_point_scatter": True,
                "output_shape": output_shape,
            }

        # Collect all variables with their coefficients
        all_vars = []
        for var in used_iter_vars:
            coeff = get_coefficient(var)
            if coeff > 0 and var in self.range_tree_nodes:
                try:
                    length = int(self.range_tree_nodes[var].length)
                    all_vars.append((str(var), coeff, length))
                except (TypeError, ValueError):
                    return None

        # Add indirect variable
        all_vars.append((indirect_var, indirect_coeff, -1))  # -1 marks as indirect

        # Sort by coefficient descending (larger coeff = earlier dimension)
        all_vars.sort(key=lambda x: x[1], reverse=True)

        # Find position of indirect variable
        indirect_pos = None
        for i, (name, coeff, length) in enumerate(all_vars):
            if name == indirect_var:
                indirect_pos = i
                break

        if indirect_pos is None:
            return None

        # Split into before and after
        dims_before = [
            (name, length) for name, coeff, length in all_vars[:indirect_pos]
        ]
        dims_after = [
            (name, length) for name, coeff, length in all_vars[indirect_pos + 1 :]
        ]

        # Verify coefficient structure for iteration variables only
        # The indirect variable's coefficient should equal product of all following iter var sizes
        # Each iter var's coefficient should equal product of all following iter var sizes
        iter_vars_after = [
            (name, coeff, length)
            for name, coeff, length in all_vars[indirect_pos + 1 :]
        ]

        expected_coeff = 1
        for name, coeff, length in reversed(iter_vars_after):
            if coeff != expected_coeff:
                return None
            expected_coeff *= length

        # Indirect var coeff should equal expected_coeff (product of all following iter var sizes)
        if indirect_coeff != expected_coeff:
            return None

        # For vars before indirect, continue the coefficient check
        # accounting for the indirect dimension's size in the output buffer
        # We need to get the output buffer's size for the indirect dimension
        # For now, we just verify the relative ordering is correct
        # by checking each var's coeff is larger than the next

        return {
            "indirect_var": indirect_var,
            "indirect_dim": indirect_pos,
            "dims_before": dims_before,
            "dims_after": dims_after,
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
            "prod": "jnp.prod",  # CPU only - not supported in Pallas GPU (Triton) backend
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

        # Get the individual pointwise dimension sizes from range_tree_nodes
        pointwise_sizes = []
        for var, entry in sorted(
            self.range_tree_nodes.items(), key=lambda x: str(x[0])
        ):
            if not entry.prefix.startswith("r"):
                try:
                    pointwise_sizes.append(int(entry.length))
                except (TypeError, ValueError):
                    pointwise_sizes = None
                    break

        # Get the pointwise and reduction numels
        pointwise_numel = 1
        for p in pointwise_prefixes:
            if p in self.numels:
                numel = self.numels[p]
                try:
                    pointwise_numel *= int(numel)
                except (TypeError, ValueError):
                    pointwise_numel = None
                    break

        reduction_numel = 1
        for p in self.numels:
            if p.startswith("r"):
                numel = self.numels[p]
                try:
                    reduction_numel *= int(numel)
                except (TypeError, ValueError):
                    reduction_numel = None
                    break

        # Count the number of pointwise and reduction dimensions
        n_reduction_dims = sum(
            1
            for var, entry in self.range_tree_nodes.items()
            if entry.prefix.startswith("r")
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
                        if entry.prefix.startswith("r")
                    ]
                    if reduction_vars:
                        r_var = reduction_vars[0]
                        # Get the coefficient (stride) of the reduction variable
                        r_coeff = load_index.coeff(r_var)
                        try:
                            r_stride = int(r_coeff) if r_coeff != 0 else 1
                        except (TypeError, ValueError):
                            r_stride = 1
                        # Get pointwise variable
                        pw_vars = [
                            var
                            for var, entry in self.range_tree_nodes.items()
                            if not entry.prefix.startswith("r")
                        ]
                        if pw_vars:
                            pw_var = pw_vars[0]
                            pw_coeff = load_index.coeff(pw_var)
                            try:
                                pw_stride = int(pw_coeff) if pw_coeff != 0 else 1
                            except (TypeError, ValueError):
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
            if is_partial_reduction:
                # For partial reductions, we need to:
                # 1. Find which axes are reduction axes (contiguous axes whose product = reduction_numel)
                # 2. Move pointwise axes to front, reduction axes to back
                # 3. Reshape to (pointwise_numel, reduction_numel) and reduce over last axis
                # 4. Reshape output with 1s in reduced dims for proper broadcasting
                reduction_op = reduction_ops[reduction_type]
                # Use a helper to find reduction axes by product matching
                reduction_expr = f"_pallas_partial_reduce({reduction_op}, {value}, {pointwise_numel}, {reduction_numel})"
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

        # For GPU (Triton backend), import pltriton for masked loads/stores
        # Import math for masked ops and symbolic expressions (e.g., math.floor, math.log2)
        imports = (
            """
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
"""
            + (
                "\nfrom jax.experimental.pallas import triton as pltriton"
                if not interpret_is_cpu
                else ""
            )
            + (
                "\nfrom torch._inductor.runtime.runtime_utils import next_power_of_2"
                if self.use_masked_ops
                else ""
            )
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

        # For GPU with masked ops, add block_size as keyword-only parameter
        kernel_signature = (
            f"def {kernel_name}_kernel({', '.join(full_kernel_params)}"
            + (", *, block_size" if self.use_masked_ops else "")
            + "):"
        )
        code.writeline(kernel_signature)
        with code.indent():
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
                        code.writeline(f"# Mask for {buf_name}")
                        code.writeline(f"{mask_var}_size = {matching_param}.size")
                        code.writeline(
                            f"{mask_var} = jnp.arange(block_size) < {mask_var}_size"
                        )

            # Generate iteration variables as jnp.arange arrays
            # These are used by index_expr operations like torch.arange
            # Skip on GPU with masked ops - iteration vars would create non-power-of-2 arrays
            # which are not supported by Pallas Triton backend
            if self.range_tree_nodes and not self.use_masked_ops:
                code.writeline("# Define iteration variables as JAX arrays")
                # Get the first output buffer's shape for reshaping
                first_output_shape = None
                first_output_numel = None
                if output_params:
                    first_out_param = output_params[0]
                    first_out_buf_name = output_buffer_lookup.get(first_out_param)
                    if first_out_buf_name:
                        try:
                            buf = V.graph.get_buffer(first_out_buf_name)
                            size = buf.get_size()
                            first_output_shape = tuple(
                                int(s) if hasattr(s, "__int__") else s for s in size
                            )
                            first_output_numel = 1
                            for s in first_output_shape:
                                first_output_numel *= s
                        except Exception:
                            pass

                # Collect all iteration variable info for broadcasting shape computation
                var_items = list(self.range_tree_nodes.items())

                # Count vars that are NOT the "total" var (which equals output numel)
                # These are the actual iteration dimensions that need broadcasting
                broadcast_vars = []
                total_var_idx = None
                for idx, (var_sym, entry) in enumerate(var_items):
                    try:
                        length_val = (
                            int(entry.length)
                            if hasattr(entry.length, "__int__")
                            else None
                        )
                    except (TypeError, ValueError):
                        length_val = None
                    if length_val is not None and length_val == first_output_numel:
                        total_var_idx = idx
                    else:
                        broadcast_vars.append((idx, var_sym, entry, length_val))

                num_broadcast_dims = len(broadcast_vars)

                for idx, (var_sym, entry) in enumerate(var_items):
                    var_name = str(var_sym)
                    length = entry.length
                    length_str = self.kexpr(length)
                    try:
                        length_val = int(length) if hasattr(length, "__int__") else None
                    except (TypeError, ValueError):
                        length_val = None

                    # Skip symbolic lengths - jnp.arange requires concrete values
                    if length_val is None:
                        continue

                    if (
                        first_output_shape
                        and len(first_output_shape) > 1
                        and length_val == first_output_numel
                    ):
                        # This is the "total" variable - reshape to output shape
                        shape_str = ", ".join(str(s) for s in first_output_shape)
                        code.writeline(
                            f"{var_name} = jnp.arange({length_str}).reshape({shape_str})"
                        )
                    elif num_broadcast_dims > 1 and idx != total_var_idx:
                        # Find position of this var among broadcast vars
                        broadcast_idx = next(
                            i
                            for i, (vidx, _, _, _) in enumerate(broadcast_vars)
                            if vidx == idx
                        )
                        # Reshape for broadcasting with other iteration vars
                        # Order: outermost to innermost should match the output shape
                        # Reverse the order so first var (smallest index) is innermost
                        # and last var (largest index) is outermost
                        reversed_idx = num_broadcast_dims - 1 - broadcast_idx
                        shape_parts = ["1"] * num_broadcast_dims
                        shape_parts[reversed_idx] = length_str
                        shape_str = ", ".join(shape_parts)
                        code.writeline(
                            f"{var_name} = jnp.arange({length_str}).reshape({shape_str})"
                        )
                    else:
                        code.writeline(f"{var_name} = jnp.arange({length_str})")

            # Emit compute (CSE) and store lines; they reference *_ptr[index] directly.
            for line in self.compute._lines:
                code.writeline(str(line))
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

            # For masked ops, calculate block_size as next power of 2 of max flattened size
            if self.use_masked_ops:
                code.writeline(
                    "# Calculate block_size as next power of 2 for Triton backend"
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
                code.writeline("block_size = next_power_of_2(max_size)")

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
                    else:
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach().contiguous())"
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
