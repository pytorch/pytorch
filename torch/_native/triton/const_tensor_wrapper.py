import torch


class ConstTensorWrapper:
    """Read-only, zero-copy view of a tensor for Triton kernel arguments.

    Triton extracts a kernel pointer argument by duck typing: any object with a
    ``data_ptr()`` method and a ``dtype`` is treated as a pointer (see Triton's
    ``jit.py``). ``ConstTensorWrapper`` overrides ``data_ptr()`` to return
    ``const_data_ptr()``, so passing a copy-on-write tensor to an argument the
    kernel only reads does not materialize it. The remaining attributes Triton
    inspects during specialization and launch are forwarded to the wrapped
    tensor.

    Only wrap arguments the kernel treats as read-only (loads, never stores).
    Triton does not enforce this; wrapping a written argument is a bug.

    Accessors run under DisableTorchFunctionSubclass so the wrapper never
    re-enters __torch_function__: const_data_ptr() and friends are dispatched
    methods, and re-entering while an override is mid-redispatch trips the
    "cannot skip two levels of __torch_function__" guard.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def data_ptr(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            # const_data_ptr() exists at runtime but is absent from the stubs.
            return self._tensor.const_data_ptr()  # type: ignore[attr-defined]

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    def dim(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.dim()

    def size(self, dim: int | None = None) -> torch.Size | int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.size() if dim is None else self._tensor.size(dim)

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.stride() if dim is None else self._tensor.stride(dim)

    def element_size(self) -> int:
        with torch._C.DisableTorchFunctionSubclass():
            return self._tensor.element_size()
