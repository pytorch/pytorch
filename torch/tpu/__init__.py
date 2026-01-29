"""TPU tensor factory functions.

Provides factory functions that allocate tensors as JAX arrays on TPU,
wrapped in a TPUTensor subclass. These tensors carry PyTorch-compatible
metadata (shape, stride, dtype) for torch.compile tracing, while the
actual data lives on TPU as JAX arrays.

Usage:
    import torch.tpu

    x = torch.tpu.empty((16,), dtype=torch.float32)
    y = torch.tpu.zeros((4, 4))

    # Use within torch.compile:
    @torch.compile(backend="inductor")
    def fn(a, b):
        return (a + b) * 2.0

    result = fn(x, y)  # Executes natively on TPU via Pallas
"""

from typing import Any, Sequence, Union

import torch

from torch.tpu._tensor import _contiguous_strides, TPUTensor


__all__ = [
    "TPUTensor",
    "empty",
    "zeros",
    "ones",
    "empty_strided",
    "rand_strided",
]

# Type alias for shape arguments
_ShapeType = Union[Sequence[int], torch.Size]


def _torch_dtype_to_jax(dtype: torch.dtype) -> Any:
    """Convert a PyTorch dtype to a JAX dtype."""
    from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime

    return torch_dtype_to_jax_runtime(dtype)


def empty(size: _ShapeType, *, dtype: torch.dtype = torch.float32) -> TPUTensor:
    """Allocate an uninitialized TPU tensor.

    Creates a JAX array via jnp.empty() explicitly on the first TPU device
    using jax.default_device, then wraps it in a TPUTensor.

    Args:
        size: Shape of the tensor.
        dtype: PyTorch dtype (default: torch.float32).

    Returns:
        TPUTensor wrapping a JAX array on TPU.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    jax_dtype = _torch_dtype_to_jax(dtype)
    with jax.default_device(jax.devices("tpu")[0]):
        jax_array = jnp.empty(size, dtype=jax_dtype)
    return TPUTensor(jax_array, dtype)


def zeros(size: _ShapeType, *, dtype: torch.dtype = torch.float32) -> TPUTensor:
    """Allocate a zero-filled TPU tensor.

    Args:
        size: Shape of the tensor.
        dtype: PyTorch dtype (default: torch.float32).

    Returns:
        TPUTensor wrapping a zero-filled JAX array on TPU.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    jax_dtype = _torch_dtype_to_jax(dtype)
    with jax.default_device(jax.devices("tpu")[0]):
        jax_array = jnp.zeros(size, dtype=jax_dtype)
    return TPUTensor(jax_array, dtype)


def ones(size: _ShapeType, *, dtype: torch.dtype = torch.float32) -> TPUTensor:
    """Allocate a ones-filled TPU tensor.

    Args:
        size: Shape of the tensor.
        dtype: PyTorch dtype (default: torch.float32).

    Returns:
        TPUTensor wrapping a ones-filled JAX array on TPU.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    jax_dtype = _torch_dtype_to_jax(dtype)
    with jax.default_device(jax.devices("tpu")[0]):
        jax_array = jnp.ones(size, dtype=jax_dtype)
    return TPUTensor(jax_array, dtype)


def empty_strided(
    size: _ShapeType,
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> TPUTensor:
    """Allocate an uninitialized TPU tensor with specified stride.

    Used by Inductor wrapper codegen for intermediate buffer allocation.
    JAX arrays are always contiguous, so stride must be contiguous.

    Args:
        size: Shape of the tensor.
        stride: Stride of the tensor (must be contiguous).
        dtype: PyTorch dtype (default: torch.float32).

    Returns:
        TPUTensor wrapping a JAX array on TPU.

    Raises:
        AssertionError: If stride is not contiguous.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    expected_stride = _contiguous_strides(tuple(size))
    assert tuple(stride) == expected_stride, (
        f"TPU tensors must be contiguous. Got stride {tuple(stride)}, "
        f"expected {expected_stride} for shape {tuple(size)}"
    )

    jax_dtype = _torch_dtype_to_jax(dtype)
    with jax.default_device(jax.devices("tpu")[0]):
        jax_array = jnp.empty(size, dtype=jax_dtype)
    return TPUTensor(jax_array, dtype)


def rand_strided(
    size: _ShapeType,
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> TPUTensor:
    """Allocate a random TPU tensor with specified stride.

    Used by generated benchmark_compiled_module to create TPUTensor inputs
    for benchmarking compiled Pallas kernels. Stride must be contiguous.

    Args:
        size: Shape of the tensor.
        stride: Stride of the tensor (must be contiguous).
        dtype: PyTorch dtype (default: torch.float32).

    Returns:
        TPUTensor wrapping a random JAX array on TPU.

    Raises:
        AssertionError: If stride is not contiguous.
    """
    import jax  # pyrefly: ignore [import-error, missing-import]
    import jax.numpy as jnp  # pyrefly: ignore [import-error, missing-import]

    expected_stride = _contiguous_strides(tuple(size))
    assert tuple(stride) == expected_stride, (
        f"TPU tensors must be contiguous. Got stride {tuple(stride)}, "
        f"expected {expected_stride} for shape {tuple(size)}"
    )

    jax_dtype = _torch_dtype_to_jax(dtype)
    with jax.default_device(jax.devices("tpu")[0]):
        key = jax.random.PRNGKey(0)
        if jnp.issubdtype(jax_dtype, jnp.floating) or jnp.issubdtype(jax_dtype, jnp.complexfloating):
            jax_array = jax.random.normal(key, shape=size, dtype=jax_dtype)
        elif jnp.issubdtype(jax_dtype, jnp.integer):
            jax_array = jax.random.randint(key, shape=size, minval=0, maxval=100, dtype=jax_dtype)
        else:
            # bool and other types: use zeros
            jax_array = jnp.zeros(size, dtype=jax_dtype)
    return TPUTensor(jax_array, dtype)
