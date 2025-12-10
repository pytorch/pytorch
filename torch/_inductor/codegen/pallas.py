from __future__ import annotations

import hashlib
import math
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet
from torch.utils._pallas import has_tpu_pallas

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

        idx_str = V.kernel.kexpr(V.kernel.prepare_indexing(expr))
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
                # Strided multi-dimensional access - requires advanced indexing
                # For now, use ellipsis which may work for many cases
                # TODO: Implement proper multi-dimensional strided indexing
                return "..."

        # For complex cases, raise an error
        return self._generate_index_array(index)

    def _generate_index_array(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for complex indexing patterns.

        For very complex patterns that can't be expressed as simple slices,
        we need to compute the indices explicitly. This is not yet fully implemented.
        """
        # For now, raise an error for complex patterns
        # TODO: Implement advanced indexing support
        raise Unsupported(
            f"Pallas backend does not yet support complex indexing pattern: {index}"
        )

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
            return self._get_index_str(index), False

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

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:  # type: ignore[override]
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        # Track the load index expression for argmax/argmin axis detection
        self.load_index_exprs[name] = index

        # Determine masked ops strategy on first load/store if not yet determined
        if self.use_masked_ops is None:
            self.use_masked_ops = self._determine_masked_ops_for_kernel()

        index_str, needs_flatten = self._get_index_expr(index)

        # Build load expression using string concatenation
        use_masked = index_str == "..." and not needs_flatten and self.use_masked_ops

        if use_masked:
            # GPU masked load: flatten tensor and apply per-tensor mask
            mask_var = self._get_or_create_mask(name)
            load_expr = f"pltriton.load({buf}.at[pl.ds(block_size)], mask={mask_var})"
        elif needs_flatten:
            # Flatten then index for non-contiguous access
            load_expr = f"{buf}[...].flatten()[{index_str}]"
        else:
            # Direct indexing for contiguous access
            load_expr = f"{buf}[{index_str}]"

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

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:  # type: ignore[override]
        if mode is not None:
            raise Unsupported("pallas store mode not supported")
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
            index_str, needs_flatten = self._get_index_expr(index)

            # Build store expression using string concatenation
            use_masked = (
                index_str == "..." and not needs_flatten and self.use_masked_ops
            )

            if use_masked:
                # GPU masked store: flatten tensor and apply per-tensor mask
                mask_var = self._get_or_create_mask(name)
                store_expr = f"pltriton.store({out}.at[pl.ds(block_size)], {value}, mask={mask_var})"
            elif index_str == "...":
                # When storing the full array, reshape to match the output shape.
                # This handles:
                # - Mixed indexing producing flat results needing reshape
                # - Squeeze operations where value has more dims than output
                # - If shapes already match, reshape is a no-op.
                # Use the output array's shape at runtime to avoid issues with
                # symbolic sizes not being defined in the kernel.
                store_expr = f"{out}[...] = {value}.reshape({out}.shape)"
            else:
                # Direct indexed assignment
                store_expr = f"{out}[{index_str}] = {value}"

        self.stores.writeline(store_expr)
        # Track which output param this store uses for filtering in codegen_kernel
        self.store_with_output.append((out, store_expr))

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
            # For argmax/argmin, we need to preserve the axis information
            # because the result is indices, not values.
            reduction_op = reduction_ops[reduction_type]
            # Check if this is a true partial reduction (pointwise numel > 1)
            # When pointwise_numel == 1, it's effectively a full reduction to scalar
            is_partial_reduction = (
                has_pointwise and pointwise_numel and pointwise_numel > 1
            )
            if is_partial_reduction and n_reduction_dims > 0:
                # Partial reduction: determine the reduction axis from load index
                # The reduction variable's coefficient in the index expression tells us its stride
                # Higher stride = outer axis (lower axis number in row-major order)
                reduction_axis = 0  # Default to axis 0
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
                            reduction_axis = 0 if r_stride > pw_stride else 1
                if n_reduction_dims == 1:
                    reduction_expr = f"{reduction_op}({value}, axis={reduction_axis})"
                else:
                    # Multiple reduction dims - reduce over all of them
                    axes = tuple(range(n_reduction_dims))
                    reduction_expr = f"{reduction_op}({value}, axis={axes})"
            else:
                # Full reduction to scalar
                reduction_expr = f"{reduction_op}({value})"
        elif reduction_type in reduction_ops:
            if (
                has_pointwise
                and pointwise_numel
                and reduction_numel
                and pointwise_sizes
            ):
                # For partial reductions, we need to:
                # 1. Move pointwise axes to the front and reduction axes to the back
                # 2. Reshape to (pointwise_numel, reduction_numel)
                # 3. Reduce over the last axis
                #
                # We use moveaxis to reorder: first move axes matching pointwise sizes
                # to the front, then the remaining (reduction) axes go to the back.
                # Finally reshape and reduce.
                #
                # Generate code to dynamically determine and reorder axes:
                pw_sizes_str = str(pointwise_sizes)
                reduction_op = reduction_ops[reduction_type]
                reduction_expr = (
                    f"(lambda v: (lambda pw_sizes: "
                    f"{reduction_op}(v.reshape(-1, {reduction_numel}), axis=-1) "
                    f"if v.ndim == 2 else "
                    f"(lambda input_shape, pw_axes: "
                    f"{reduction_op}("
                    f"jnp.moveaxis(v, pw_axes, list(range(len(pw_axes)))).reshape({pointwise_numel}, -1), axis=-1)"
                    f")("
                    f"v.shape, "
                    f"[i for i, s in enumerate(v.shape) if s in pw_sizes][:len(pw_sizes)]"
                    f")"
                    f")({pw_sizes_str}))({value})"
                )
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
        arg_defs, _, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_params if p.startswith("out_ptr")]
        output_params = [
            p for p in kernel_params if p.startswith(("out_ptr", "in_out_ptr"))
        ]
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
        # Import math at module level if we'll use it for masked ops
        imports = (
            """
            import functools
            """
            + ("import math\n            " if self.use_masked_ops else "")
            + """import torch
            import jax
            import jax.numpy as jnp
            from jax.experimental import pallas as pl
            from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime
            """
            + (
                "\n            from jax.experimental.pallas import triton as pltriton"
                if not interpret_is_cpu
                else ""
            )
            + (
                "\n            from torch._inductor.runtime.runtime_utils import next_power_of_2"
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
            aliasable_flags[param] = (not interpret_is_cpu) and is_contiguous
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
        copy_output_indices = [
            idx for idx, name in enumerate(output_params) if name in non_alias_out_set
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

                for var_sym, entry in self.range_tree_nodes.items():
                    var_name = str(var_sym)
                    length = entry.length
                    length_str = self.kexpr(length)
                    # If the iteration variable length matches the output numel,
                    # reshape it to match the output shape for proper broadcasting
                    try:
                        length_val = int(length) if hasattr(length, "__int__") else None
                    except (TypeError, ValueError):
                        length_val = None

                    # Skip symbolic lengths - jnp.arange requires concrete values
                    # This happens with dynamic shapes
                    if length_val is None:
                        continue

                    if (
                        first_output_shape
                        and len(first_output_shape) > 1
                        and length_val == first_output_numel
                    ):
                        shape_str = ", ".join(str(s) for s in first_output_shape)
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
        for idx, name in enumerate(kernel_input_params):
            if (name in alias_params) or name.startswith("in_out_ptr"):
                donate_indices.append(idx + 2)
        if donate_indices:
            donate_literal = "(" + ", ".join(str(x) for x in donate_indices) + ",)"
        else:
            donate_literal = "()"
        code.writeline(
            "@functools.partial("
            "jax.jit, static_argnums=(0, 1), donate_argnums="
            f"{donate_literal})"
        )
        code.writeline(
            f"def {jit_wrapper_name}(out_shapes, out_dtypes, {', '.join(kernel_input_params)}):"
        )
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

            # For masked ops, wrap kernel with functools.partial to pass block_size
            kernel_arg = (
                f"functools.partial({kernel_name}_kernel, block_size=block_size),"
                if self.use_masked_ops
                else f"{kernel_name}_kernel,"
            )
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

            if kernel_input_params:
                alias_args_str = ", ".join(
                    arg_name_map[name] for name in kernel_input_params
                )
                code.writeline(
                    f"res = {jit_wrapper_name}(out_shapes, out_dtypes, {alias_args_str})"
                )
            else:
                code.writeline(f"res = {jit_wrapper_name}(out_shapes, out_dtypes)")
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
