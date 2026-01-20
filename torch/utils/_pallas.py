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
    """Check if JAX has CUDA backend support with SM90+ (required by Mosaic GPU)."""
    if not has_jax_package():
        return False
    try:
        import jax  # type: ignore[import-not-found]

        # Check if CUDA backend is available
        devices = jax.devices("gpu")
        if len(devices) == 0:
            return False

        # Mosaic GPU requires SM90+ (compute capability 9.0+)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major < 9:
                return False

        return True
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
def has_cpu_pallas() -> bool:
    """Checks for a full Pallas-on-CPU environment."""
    return has_pallas_package()


@functools.cache
def has_cuda_pallas() -> bool:
    """Checks for a full Pallas-on-CUDA environment."""
    return has_pallas_package() and torch.cuda.is_available() and has_jax_cuda_backend()


@functools.cache
def has_tpu_pallas() -> bool:
    """Checks for a full Pallas-on-TPU environment."""
    return has_pallas_package() and has_jax_tpu_backend()


@functools.cache
def has_pallas() -> bool:
    """
    Check if Pallas backend is fully available for use.

    Requirements:
    - JAX package installed
    - Pallas (jax.experimental.pallas) available
    - A compatible backend (CUDA or TPU) is available in both PyTorch and JAX.
    """
    return has_cpu_pallas() or has_cuda_pallas() or has_tpu_pallas()
