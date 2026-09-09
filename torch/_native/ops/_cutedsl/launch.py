# Shared CuteDSL launch glue: the operand DESCRIPTORS a kernel is compiled against and the
# tvm-ffi launcher, reused by every CuteDSL native op.
#
# A kernel is compiled against FAKE operands and the compiled callable then takes the torch
# tensors themselves, so there is no per-call wrap -- which is where the host time went:
# 5.98us/call against 19.16 for wrapping each operand every call.
#
# The STREAM stays an explicit argument. tvm-ffi's ENV stream cannot serve here -- its detector
# needs a top-level GPU tensor argument and these kernels pass operands as lists -- and the
# packed-id form of the query deadlocks capture. INPUTS go through read_only(): a
# copy-on-write input must export via const_data_ptr() or it is silently MATERIALIZED, which
# the autograd backward contract forbids under a transparent override.

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass.cute as cute

import torch
from torch.utils.dlpack import ReadOnlyTensorWrapper


def sym(divisibility: int = 1):
    """A DYNAMIC extent or stride, guaranteed divisible by `divisibility`.

    One compiled kernel then serves every value sharing that divisor, and the divisor is
    what lets the kernel keep emitting wide loads.
    """
    return cute.sym_int(divisibility=divisibility)


def fake_compact(dtype, shape, *, order=None, align=None):
    """Compile-time descriptor for a COMPACT operand.

    `order` lists the modes fastest-varying LAST. `align` is what the kernel may assume, and
    the caller must have checked the real pointer meets it -- a broken claim faults at launch.
    """
    return cute.runtime.make_fake_compact_tensor(
        dtype, tuple(shape), stride_order=order, assumed_align=align
    )


def read_only(t):
    """Wrap an INPUT so it exports through const_data_ptr(), leaving a COW input unmaterialized.

    Inputs only -- outputs must stay writable -- and it rejects every non-DLPack op, so wrap
    the final-shape tensor.
    """
    return ReadOnlyTensorWrapper(t)


def compile_kernel(op, *args):
    """Compile `op` against FAKE operands, for the fast tvm-ffi arg convention."""
    return cute.compile(op, *args, options="--enable-tvm-ffi")


def stream():
    # Live current stream handle, read every call and never cached: callers may set a different
    # stream or device per call, including the CUDA-graph capture stream.
    return cuda.CUstream(
        torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
    )
