# Shared CuteDSL launch glue: the dtype map, the operand DESCRIPTORS a kernel is compiled against,
# and the tvm-ffi launcher. Reused by every CuteDSL native op.
#
# A kernel is compiled against FAKE operands -- dtype, extents (static or symbolic), strides,
# declared alignment -- and the compiled callable then takes the torch tensors themselves. There is
# no per-call wrap at all, which is where the host time went: MEASURED on H100 over a two-operand
# kernel, 5.98us/call against 19.16 for wrapping each operand on every call. It is also the shape the
# vendored reference kernels and the RNG / topk / scatter_add families already use.
#
# The STREAM stays an explicit argument: _cuda_getCurrentRawStream gives the raw cudaStream_t in
# ~0.07us and tracks the graph-capture stream. The tvm-ffi ENV stream cannot serve here -- its
# detector needs a top-level GPU tensor argument, and these kernels pass their operands as lists.
# NOT _cuda_getCurrentStream(dev)[0], a packed id rather than a pointer, which deadlocks capture.
#
# INPUTS go through read_only(). A copy-on-write input has to export via const_data_ptr() or it is
# MATERIALIZED, which the autograd backward contract forbids under a transparent override -- and
# passing the bare tensor does materialize it, silently.

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass.cute as cute

import torch
from torch.utils.dlpack import ReadOnlyTensorWrapper


def sym(divisibility: int = 1):
    """A DYNAMIC extent or stride, guaranteed divisible by `divisibility`.

    One compiled kernel then serves every value sharing that divisor -- the vec class -- instead of
    one kernel per distinct shape. The divisor is what lets the kernel keep emitting wide loads.
    """
    return cute.sym_int(divisibility=divisibility)


def fake_compact(dtype, shape, *, order=None, align=None):
    """Compile-time descriptor for a COMPACT operand.

    `order` lists the modes fastest-varying LAST, so (1, 0) is a row-major 2D tensor and (2, 1, 0) a
    row-major 3D one; None takes the DSL default. `align` is the alignment the kernel may assume:
    declaring it is not optional for a wide load, and the caller must have checked that the real base
    pointer meets it -- a declared claim the pointer breaks faults at launch.
    """
    return cute.runtime.make_fake_compact_tensor(
        dtype, tuple(shape), stride_order=order, assumed_align=align
    )


def fake_strided(dtype, shape, stride, *, align=None):
    """Compile-time descriptor for a GAPPED operand: a dense run per row, rows further apart than
    the run. Strides may be symbolic (see `sym`), which is how one kernel serves every row pitch.
    """
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(stride), assumed_align=align
    )


def read_only(t):
    """Wrap an INPUT so it exports through const_data_ptr().

    A copy-on-write input is then NOT materialized. Apply to inputs only -- outputs are written by
    the kernel and must stay writable. The wrapper rejects every non-DLPack op, so it wraps the
    final-shape tensor, after any reshape the caller did itself.
    """
    return ReadOnlyTensorWrapper(t)


def compile_kernel(op, *args):
    """Compile `op` against FAKE operands, for the fast tvm-ffi arg convention.

    The flag is equivalent to the cute.compile[EnableTVMFFI] typed form (measured identical host
    dispatch) and matches the convention the other native CuteDSL ops use.
    """
    return cute.compile(op, *args, options="--enable-tvm-ffi")


def stream():
    # Live current stream handle, read every call (never cached: callers may set a different
    # stream/device per call, and the kernel must launch on whatever is current, including the
    # CUDA-graph capture stream).
    return cuda.CUstream(
        torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
    )
