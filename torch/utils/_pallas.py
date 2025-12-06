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
def has_jax_tpu_backend() -> bool:
    """Check if JAX has TPU backend support."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if TPU backend is available
        devices = jax.devices("tpu")
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
    - A compatible backend (CUDA or TPU) is available in both PyTorch and JAX.
    """
    if not has_pallas_package():
        return False

    # Check for is CUDA is available or if JAX has GPU/CUDA backend
    has_cuda = torch.cuda.is_available() and has_jax_cuda_backend()

    # Check for TPU backend
    has_tpu_torch = False
    try:
        import torch_xla.core.xla_model as xm

        has_tpu_torch = xm.xla_device_count() > 0
    except ImportError:
        pass
    has_tpu = has_tpu_torch and has_jax_tpu_backend()

    return has_cuda or has_tpu
