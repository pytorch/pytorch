"""
Python override for aten::linalg_polar (polar decomposition A = U @ H).

Dispatch:

  CUDA, nvmath available (cuSOLVER Xpolar; cuSOLVER >= 12.2, i.e. the
  cusolverDnXpolar symbol introduced in CUDA 13.2)?
  |
  |-- YES --> QR-based Dynamically Weighted Halley (QDWH) via cuSOLVER Xpolar
  |
  |-- NO  --> SVD-based polar decomposition (runs on the input's device)

The CPU override always uses the SVD path; the CUDA path uses cuSOLVER QDWH when
available and otherwise SVD on the CUDA tensor. Both keys share a cond that
matches only supported dtypes and m >= n matrices, so unsupported inputs fall
through to the structured aten kernel (whose meta function raises the
appropriate shape/dtype error) instead of silently running the SVD path.
"""

import warnings

import torch

from ... import registry
from ...common_utils import _unavailable_reason


# Dtypes the op supports overall (the SVD path handles all of these on any
# device). cuSOLVER Xpolar only supports real types, so complex inputs always
# take the SVD path even on CUDA -- see _XPOLAR_DTYPES.
_SUPPORTED_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

# Dtypes for which the cuSOLVER Xpolar (QDWH) fast path is attempted.
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
# permanently if a call ever raises (e.g. runtime lacks the symbol).
_nvmath_available: "bool | None" = None
_nvmath_warned = False


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
        # device.type == "cuda"). ROCm falls through to the SVD path.
        version = None if torch.version.hip is not None else _cusolver_version()
        _nvmath_available = (
            torch.version.hip is None
            and _unavailable_reason(_NVMATH_DEPS) is None
            and version is not None
            and version >= _MIN_CUSOLVER_VERSION
        )
    return _nvmath_available


def _polar_cond(A: torch.Tensor) -> bool:
    # Match only inputs we support: 2-D+ matrices with m >= n and a supported
    # dtype. On a mismatch the router falls through to the aten kernel, whose
    # meta function raises the proper error (e.g. "must have at least as many
    # rows as columns"). This keeps CPU and CUDA error behavior identical.
    if A.dtype not in _SUPPORTED_DTYPES:
        return False
    return A.dim() >= 2 and A.size(-2) >= A.size(-1)


def _polar_impl_svd(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Polar decomposition from the (reduced) SVD: A = U_ diag(S) Vh, so the
    # orthogonal factor is U = U_ @ Vh and H = Vh^H diag(S) Vh (Hermitian PSD).
    U_, S, Vh = torch.linalg.svd(A, full_matrices=False)
    U = U_ @ Vh
    H = Vh.mH @ (S.unsqueeze(-1) * Vh)
    # Symmetrize/Hermitianize to clean up round-off.
    H = 0.5 * (H + H.mH)
    return U, H


def _polar_impl_cuda(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global _nvmath_available
    if A.dtype in _XPOLAR_DTYPES and _check_nvmath():
        from .nvmath_impl import polar_xpolar

        try:
            U, H, _ = polar_xpolar(A)
            return U, H
        except Exception:
            # Hard failure (missing symbol, bad runtime); stop trying Xpolar
            # and fall back to SVD for this and subsequent calls.
            _nvmath_available = False
    elif A.dtype not in _XPOLAR_DTYPES:
        pass  # complex: Xpolar unsupported, use SVD
    else:
        global _nvmath_warned
        if not _nvmath_warned:
            _nvmath_warned = True
            reason = _unavailable_reason(_NVMATH_DEPS) or (
                "cuSOLVER >= 12.2 (CUDA 13.2) runtime with Xpolar not found"
            )
            warnings.warn(
                f"linalg.polar: cuSOLVER QDWH path unavailable ({reason}), "
                f"using slower SVD-based fallback.",
                stacklevel=3,
            )
    return _polar_impl_svd(A)


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar",
        "CUDA",
        cond=_polar_cond,
        impl=_polar_impl_cuda,
    )
    registry.register_op_override(
        "native",
        "aten",
        "linalg_polar",
        "CPU",
        cond=_polar_cond,
        impl=_polar_impl_svd,
    )
