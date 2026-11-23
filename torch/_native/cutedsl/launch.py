# Shared operand descriptors and tvm-ffi launch glue. Compile against fake operands, then pass
# torch tensors directly (5.98us versus 19.16us with per-call wrapping). Keep the stream explicit:
# ENV detection misses tensors inside lists and packed-id lookup deadlocks capture. read_only()
# exports COW inputs through const_data_ptr() to avoid forbidden materialization in backward.

from collections.abc import Callable, Sequence
from typing import Any, overload, TYPE_CHECKING

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute

import torch
from torch._native.utils.tensor import const_data_ptr
from torch.utils.dlpack import ReadOnlyTensorWrapper


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


def sym(divisibility: int = 1) -> cute.SymInt:
    """A dynamic extent or stride divisible by divisibility, allowing one wide-load
    kernel to serve every matching value.
    """
    return cute.sym_int(divisibility=divisibility)


def fake_compact(
    dtype: type[cutlass.Numeric],
    shape: Sequence[int | cute.SymInt],
    *,
    stride_order: tuple[int, ...] | None = None,
    align: int | None = None,
) -> cute.Tensor:
    """Describe a compact compile-time operand. `stride_order` gives each mode's
    compactness rank (0 is stride-1); the pointer must satisfy `align`.
    """
    return cute.runtime.make_fake_compact_tensor(
        dtype, tuple(shape), stride_order=stride_order, assumed_align=align
    )


def supported_alignment(tensor: torch.Tensor, maximum: int) -> int:
    """Return the largest power-of-two alignment up to `maximum` supported by `tensor`."""
    ptr = const_data_ptr(tensor)
    alignment = maximum
    while alignment > tensor.element_size() and ptr % alignment:
        alignment //= 2
    return alignment


@overload
def read_only(t: torch.Tensor) -> ReadOnlyTensorWrapper: ...


@overload
def read_only(t: None) -> None: ...


def read_only(t: torch.Tensor | None) -> ReadOnlyTensorWrapper | None:
    """Export an input through const_data_ptr() without materializing COW storage.
    Outputs must remain writable. Non-DLPack operations are rejected, so wrap only
    the final-shape tensor.
    """
    if t is None:
        return None
    with torch._C.DisableTorchFunctionSubclass():
        return ReadOnlyTensorWrapper(t)


def compile_kernel(op: Callable[..., Any], *args: Any, options: str = "") -> "Function":
    """Compile `op` against FAKE operands, for the fast tvm-ffi arg convention."""
    return cute.compile(op, *args, options=f"--enable-tvm-ffi {options}".rstrip())


def stream(device_index: int | None = None) -> cuda.CUstream:
    # Read the live stream each call because callers can change device or capture stream.
    if device_index is None:
        device_index = torch.cuda.current_device()
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(device_index))
