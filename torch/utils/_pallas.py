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


import sys

def has_torch_xla_device() -> bool:
    try:
        print("DEBUG: Attempting to import torch_xla.core.xla_model", file=sys.stderr)
        import torch_xla.core.xla_model as xm
        print("DEBUG: Import successful", file=sys.stderr)
        world_size = xm.xrt_world_size()
        print(f"DEBUG: xm.xrt_world_size() returned: {world_size}", file=sys.stderr)
        return world_size > 0
    except ImportError as e:
        print(f"DEBUG: ImportError caught: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"DEBUG: An unexpected error occurred: {e}", file=sys.stderr)
        return False


@functools.cache
def has_tpu_pallas() -> bool:
    """Checks for a full Pallas-on-TPU environment."""
    return (
        has_pallas_package()
        and has_torch_xla_device()
        and has_jax_tpu_backend()
    )


@functools.cache
def has_cuda_pallas() -> bool:
    """Checks for a full Pallas-on-CUDA environment."""
    return (
        has_pallas_package()
        and torch.cuda.is_available()
        and has_jax_cuda_backend()
    )


@functools.cache
def has_pallas() -> bool:
    """
    Check if Pallas backend is fully available for use.

    Requirements:
    - JAX package installed
    - Pallas (jax.experimental.pallas) available
    - A compatible backend (CUDA or TPU) is available in both PyTorch and JAX.
    """
    return has_cuda_pallas() or has_tpu_pallas()
