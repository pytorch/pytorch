# Shared operand descriptors and tvm-ffi launch glue. Compile against fake operands, then pass
# torch tensors directly (5.98us versus 19.16us with per-call wrapping). Keep the stream explicit:
# ENV detection misses tensors inside lists and packed-id lookup deadlocks capture. read_only()
# exports COW inputs through const_data_ptr() to avoid forbidden materialization in backward.

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass.cute as cute

import torch
from torch.utils.dlpack import ReadOnlyTensorWrapper


def sym(divisibility: int = 1):
    """A dynamic extent or stride divisible by divisibility, allowing one wide-load
    kernel to serve every matching value.
    """
    return cute.sym_int(divisibility=divisibility)


def fake_compact(dtype, shape, *, stride_order=None, align=None):
    """Describe a compact compile-time operand. `stride_order` gives each mode's
    compactness rank (0 is stride-1); the pointer must satisfy `align`.
    """
    return cute.runtime.make_fake_compact_tensor(
        dtype, tuple(shape), stride_order=stride_order, assumed_align=align
    )


def read_only(t):
    """Export an input through const_data_ptr() without materializing COW storage.
    Outputs must remain writable. Non-DLPack operations are rejected, so wrap only
    the final-shape tensor.
    """
    return ReadOnlyTensorWrapper(t)


def compile_kernel(op, *args):
    """Compile `op` against FAKE operands, for the fast tvm-ffi arg convention."""
    return cute.compile(op, *args, options="--enable-tvm-ffi")


def stream():
    # Read the live stream each call because callers can change device or capture stream.
    return cuda.CUstream(
        torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
    )
