import functools

import torch


@functools.cache
def has_jax_package() -> bool:
    """Check if JAX is installed."""
    try:
        import jax  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


@functools.cache
def has_pallas_package() -> bool:
    """Check if Pallas (JAX experimental) is available."""
    if not has_jax_package():
        return False
    try:
        from jax.experimental import (  # noqa: F401  # type: ignore[import-not-found]
            pallas as pl,
        )

        return True
    except ImportError:
        return False


@functools.cache
def get_jax_version(fallback: tuple[int, int, int] = (0, 0, 0)) -> tuple[int, int, int]:
    """Get JAX version as (major, minor, patch) tuple."""
    try:
        import jax  # type: ignore[import-not-found]

        version_parts = jax.__version__.split(".")
        major, minor, patch = (int(v) for v in version_parts[:3])
        return (major, minor, patch)
    except (ImportError, ValueError, AttributeError):
        return fallback


@functools.cache
def has_jax_cuda_backend() -> bool:
    """Check if JAX has CUDA backend support."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if CUDA backend is available
        devices = jax.devices("gpu")
        return len(devices) > 0
    except Exception:
        return False


@functools.cache
def has_pallas() -> bool:
    """
    Check if Pallas backend is fully available for use.

    Requirements:
    - JAX package installed
    - Pallas (jax.experimental.pallas) available
    - CUDA backend available (for GPU support)
    """
    if not has_pallas_package():
        return False

    # Only enable Pallas if CUDA is available
    # (Pallas primarily targets GPU workloads)
    if not torch.cuda.is_available():
        return False

    # Check if JAX has GPU/CUDA backend
    if not has_jax_cuda_backend():
        return False

    return True
