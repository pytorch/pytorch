"""TPUTensor: A tensor subclass wrapping a JAX TPU array.

Provides PyTorch-compatible metadata (shape, stride, dtype) while storing
actual data as a JAX array on TPU. Must be used within torch.compile --
eager compute ops produce wrong results (they operate on the CPU backing
storage, not the JAX array).

How it works with torch.compile:
  - Dynamo tracing: Only reads metadata (shape, dtype). Creates a FakeTensor
    for symbolic execution. The JAX array is never accessed during compilation.
    Because __torch_dispatch__ is NOT overridden, Dynamo's builder enters
    wrap_tensor() and wraps as TensorWithTFOverrideVariable.
  - AOTAutograd: Passes through without subclass decomposition (no
    __tensor_flatten__), so compiled code receives TPUTensor directly.
  - Inductor codegen: With pallas_tpu_native=True, generates code that
    accesses ._jax_array at runtime instead of DLPack conversion.
  - Runtime: Generated _main() extracts ._jax_array from inputs, runs
    Pallas kernel natively on TPU, stores result back to ._jax_array.
"""

from typing import Any, Tuple

import torch

_reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


def _contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute contiguous (row-major) strides for a given shape."""
    if not shape:
        return ()
    strides = [1]
    for s in reversed(shape[1:]):
        strides.append(strides[-1] * max(s, 1))
    strides.reverse()
    return tuple(strides)


class TPUTensor(torch.Tensor):
    """Tensor subclass wrapping a JAX TPU array.

    Provides correct PyTorch metadata (shape, stride, dtype, device='cpu')
    with zero CPU memory allocation. The actual compute data lives in
    ._jax_array as a JAX array on TPU.

    Uses _reinterpret_tensor on a zero-byte storage to create a tensor with
    correct metadata but no backing memory. This avoids allocating
    shape-proportional CPU memory (e.g. 4GB for a 1024^3 float32 tensor)
    that would never be read.

    Design choices:
      - No __torch_dispatch__ override: Dynamo's builder checks
        type(value).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
        (builder.py:764). If overridden, Dynamo skips wrap_tensor() and
        cannot trace through the subclass. By NOT overriding, Dynamo wraps
        as TensorWithTFOverrideVariable and traces ops on FakeTensor.
      - No __tensor_flatten__/__tensor_unflatten__: Prevents AOTAutograd
        from decomposing the subclass. The compiled function receives
        TPUTensor directly at runtime.
      - device='cpu' metadata: 'tpu' is not a registered PyTorch device.
        Inductor routes to Pallas backend via cpu_backend="pallas" config.
    """

    _jax_array: Any

    @staticmethod
    def __new__(cls, jax_array: Any, dtype: torch.dtype) -> "TPUTensor":
        shape = tuple(jax_array.shape)
        strides = _contiguous_strides(shape)

        # Zero-memory backing: a 0-byte CPU storage reinterpreted with the
        # desired shape/strides gives Dynamo correct metadata without
        # allocating shape-proportional CPU memory.
        base = torch.empty(0, dtype=dtype, device="cpu")
        instance = _reinterpret_tensor(base, shape, strides).as_subclass(cls)
        instance._jax_array = jax_array
        return instance

    def as_subclass(self, cls: type) -> "torch.Tensor":
        # Dynamo's output reconstruction calls to_subclass(tensor, cls) which
        # invokes tensor.as_subclass(cls). The base as_subclass creates a new
        # Python object sharing the same TensorImpl but with an empty __dict__,
        # dropping _jax_array. Override to preserve it.
        result = super().as_subclass(cls)
        result._jax_array = self._jax_array
        return result

    def __repr__(self) -> str:
        devices = self._jax_array.devices()
        device_str = ", ".join(str(d) for d in devices) if devices else "unknown"
        return (
            f"TPUTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"jax_dtype={self._jax_array.dtype}, jax_device={device_str})"
        )

    # Operations that are safe to execute on TPUTensor because they only
    # access metadata (shape, dtype, etc.), not the actual tensor data.
    _SAFE_FUNC_NAMES = frozenset({
        # Descriptor protocol - used for property access (.shape, .dtype, etc.)
        "__get__",
        "__set__",
        "__delete__",
        # Formatting - used by print/repr
        "__format__",
        "__repr__",
        "__str__",
        # Tensor metadata methods - these don't read data, only metadata
        "dim",
        "ndim",
        "size",
        "numel",
        "stride",
        "element_size",
        "is_contiguous",
        "is_complex",
        "is_floating_point",
        "is_signed",
        "is_sparse",
        "is_quantized",
        "is_meta",
        "is_nested",
        "is_inference",
        "requires_grad_",
        "storage_offset",
        "data_ptr",
        "get_device",
        # View creation metadata (doesn't read data)
        "_is_view",
        "is_leaf",
    })

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Guard against eager mode usage. TPUTensor has a zero-byte CPU backing
        # storage, so eager compute operations would read/write garbage memory.
        #
        # Why __torch_function__ and not __torch_dispatch__?
        #   - __torch_dispatch__ cannot be overridden (see NOTE below)
        #   - __torch_function__ is called at Python level before dispatch
        #
        # We allow operations when:
        #   1. Inside Dynamo tracing (is_compiling=True)
        #   2. Inside compilation pipeline (TracingContext exists)
        #   3. Safe metadata operations (property access, formatting)
        #
        # We block compute operations (add, mul, etc.) in pure eager mode.
        func_name = getattr(func, "__name__", str(func))

        # Allow safe metadata operations unconditionally
        if func_name in cls._SAFE_FUNC_NAMES:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))

        if torch._dynamo.is_compiling():
            # Inside Dynamo symbolic tracing - allow the operation to proceed
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))

        # Check if we're inside the compilation pipeline (includes variable
        # building, guard creation, AOTAutograd, etc.). TracingContext is set
        # when entering Dynamo's convert_frame and covers all compilation phases.
        from torch._guards import TracingContext

        if TracingContext.try_get() is not None:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))

        raise RuntimeError(
            f"TPUTensor does not support eager operation '{func_name}'. "
            f"TPUTensor must be used within torch.compile() with "
            f"cpu_backend='pallas' and pallas_tpu_native=True."
        )

    # NOTE: __torch_dispatch__ is intentionally NOT overridden.
    # Dynamo's builder.py:764 checks:
    #   type(value).__torch_dispatch__ is torch.Tensor.__torch_dispatch__
    # If this is True (not overridden), Dynamo calls wrap_tensor() which
    # creates TensorWithTFOverrideVariable, allowing FakeTensor tracing.
    # If we overrode __torch_dispatch__, Dynamo would skip wrap_tensor()
    # and fail to trace operations on TPUTensor inputs.

    # NOTE: __tensor_flatten__ / __tensor_unflatten__ are intentionally
    # NOT implemented. This prevents AOTAutograd from decomposing the
    # subclass (requires_subclass_dispatch() returns False), so the
    # compiled Runner.call() receives actual TPUTensor instances at
    # runtime, giving generated code access to ._jax_array.
