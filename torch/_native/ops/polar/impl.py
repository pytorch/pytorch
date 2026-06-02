"""
Python override for aten::linalg_polar (polar decomposition A = U @ H).

This override is a CUDA-only accelerator: when nvmath-python is available and
the cuSOLVER runtime exposes Xpolar, it computes the decomposition with the
QR-based Dynamically Weighted Halley (QDWH) algorithm. For every other case --
CPU, ROCm, complex inputs, missing/old nvmath, or unsupported shapes -- the
cond returns False and the router falls through to the structured aten kernel,
which computes the same decomposition via the SVD. That fallthrough is a
correct, fully-functional kernel (it also backs compile/export), so this
override never needs to provide its own CPU/SVD path.

Both the functional (linalg_polar) and out= (linalg_polar.out) overloads are
overridden so the two call forms use the same algorithm. Only single 2-D
matrices are accelerated: cuSOLVER Xpolar has no batched API, so a batched
input would require a Python loop that is slower than the batched SVD kernel;
batched inputs therefore fall through to aten.
"""

import torch

from ... import registry
from ...common_utils import _unavailable_reason


# Dtypes for which the cuSOLVER Xpolar (QDWH) fast path is attempted. Xpolar
# supports only real single/double; complex falls through to the aten SVD
# kernel. This is the single source of truth for the override's dtype gating.
_XPOLAR_DTYPES = (torch.float32, torch.float64)

_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]

# Minimum cuSOLVER runtime version exposing cusolverDnXpolar (CUDA 13.2 ships
# cuSOLVER 12.2). cusolverGetVersion encodes major*1000 + minor*100 + patch and
# does not expose the build number, so this only gates out runtimes that lack
# the symbol entirely -- it cannot distinguish patch builds.
_MIN_CUSOLVER_VERSION = 12200

# cuSOLVER Xpolar availability. Checked lazily on first use; flipped to False
# permanently if a call ever hits a hard failure (e.g. runtime lacks the symbol).
_nvmath_available: "bool | None" = None


def _cusolver_version() -> int | None:
    # Query cusolverGetVersion via nvmath against the same cuSOLVER library it
    # dlopens. Returns None if nvmath/cuSOLVER can't be loaded.
    try:
        from nvmath.bindings import cusolver  # pyrefly: ignore[missing-import]

        return cusolver.get_version()
    except Exception:
        return None


def _check_nvmath() -> bool:
    global _nvmath_available
    if _nvmath_available is None:
        # The cuSOLVER Xpolar path is NVIDIA-only; never attempt it on ROCm
        # (torch.version.hip is set for HIP builds, where tensors still report
        # device.type == "cuda"). ROCm falls through to the aten SVD kernel.
        version = None if torch.version.hip is not None else _cusolver_version()
        _nvmath_available = (
            torch.version.hip is None
            and _unavailable_reason(_NVMATH_DEPS) is None
            and version is not None
            and version >= _MIN_CUSOLVER_VERSION
        )
    return _nvmath_available


def _out_matches(A: torch.Tensor, U: torch.Tensor, H: torch.Tensor) -> bool:
    # The override writes results into U/H with copy_, which neither enforces the
    # output dtype nor resizes like the structured out= machinery does. Only fire
    # when the provided outputs already satisfy that contract (correct dtype,
    # device, and the meta-declared contiguous shapes); otherwise decline so the
    # call falls through to the aten .out kernel, which validates and resizes.
    m, n = A.shape
    return (
        U.dtype == A.dtype
        and H.dtype == A.dtype
        and U.device == A.device
        and H.device == A.device
        and tuple(U.shape) == (m, n)
        and tuple(H.shape) == (n, n)
        and U.is_contiguous()
        and H.is_contiguous()
    )


def _polar_cond_cuda(A: torch.Tensor, *args, **kwargs) -> bool:
    # Only fire when we can actually accelerate: a single non-empty real 2-D
    # matrix with m >= n, and a usable cuSOLVER Xpolar runtime. Batched inputs
    # (dim > 2) are declined -- Xpolar has no batched API, so looping in Python
    # is slower than the batched SVD kernel. Everything else (CPU, complex,
    # batched, unsupported runtime) falls through to the structured aten kernel.
    if A.dtype not in _XPOLAR_DTYPES:
        return False
    if A.dim() != 2 or A.size(0) < A.size(1) or A.size(1) == 0:
        return False
    if not _check_nvmath():
        return False
    # For the out= overload, only accelerate when the outputs already match the
    # structured contract; otherwise fall through to aten for validation/resize.
    U, H = kwargs.get("U"), kwargs.get("H")
    if U is not None or H is not None:
        if U is None or H is None or not _out_matches(A, U, H):
            return False
    return True


def _run_xpolar(A: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor] | None":
    # Returns (U, H) from cuSOLVER Xpolar, or None on a hard failure (after
    # permanently disabling the fast path).
    global _nvmath_available
    from .nvmath_impl import polar_xpolar

    try:
        U, H, _ = polar_xpolar(A)
        return U, H
    except Exception:
        # Hard failure (missing symbol, bad runtime). Disable the fast path so
        # subsequent calls (and the fall-throughs below) go straight to aten.
        _nvmath_available = False
        return None


def _polar_impl_cuda(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    result = _run_xpolar(A)
    if result is None:
        return torch.ops.aten.linalg_polar.default(A)
    # Match the meta function's contiguous output layout.
    U, H = result
    return U.contiguous(), H.contiguous()


def _polar_out_impl_cuda(
    A: torch.Tensor, *, U: torch.Tensor, H: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    result = _run_xpolar(A)
    if result is None:
        return torch.ops.aten.linalg_polar.out(A, U=U, H=H)
    U_qdwh, H_qdwh = result
    U.copy_(U_qdwh)
    H.copy_(H_qdwh)
    return U, H


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar",
        "CUDA",
        cond=_polar_cond_cuda,
        impl=_polar_impl_cuda,
    )
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar.out",
        "CUDA",
        cond=_polar_cond_cuda,
        impl=_polar_out_impl_cuda,
    )
