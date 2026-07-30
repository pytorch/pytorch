# Shared CuteDSL launch glue: torch <-> cute tensor wrapping, stream handle, and
# the EnableTVMFFI-compiled launcher. Reused by all CuteDSL native ops (reductions
# today, pointwise later) -- the host-overhead-minimizing path established by the
# pointwise/reduction experiment.
#
# Three host-overhead levers, all measured on B200 (see the experiment notes):
#   - enable_tvm_ffi=True on from_dlpack: ~0.8us vs ~3.6us for the __dlpack__()
#     capsule roundtrip (takes torch's fast C exchange).
#   - options="--enable-tvm-ffi" on cute.compile: per-compile arg-passing convention
#     that skips the per-arg get_c_pointers marshalling (~30% off the launch
#     dispatch). Equivalent to the cute.compile[EnableTVMFFI] typed form.
#   - _stream via _cuda_getCurrentRawStream: the raw cudaStream_t POINTER in ~0.07us
#     (vs ~2.7us for the torch.cuda.current_stream() wrapper) AND it correctly
#     tracks the CUDA-graph capture stream. Do NOT use _cuda_getCurrentStream(dev)[0]
#     (a packed stream id, not a pointer -- deadlocks graph capture).

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

import torch
from torch.utils.dlpack import ReadOnlyTensorWrapper


# torch dtype -> cute numeric type. Extend as new dtypes are supported.
torch2cute = {
    torch.float32: Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.int32: Int32,
}


def _ro(t, read_only):
    # Wrap an INPUT tensor read-only so the fast tvm-ffi from_dlpack exports through
    # const_data_ptr(): a copy-on-write input is exported WITHOUT materializing (the
    # autograd backward contract forbids materializing a COW view under a transparent
    # override), so the overrides no longer need a COW cond check. Apply ONLY to
    # inputs -- outputs are written by the kernel and must stay writable, so callers
    # pass read_only=False for them. The wrapper is export-only and rejects every
    # non-DLPack op, so it must wrap the FINAL-shape tensor, immediately before the
    # from_dlpack call (after any reshape/expand the helper did itself).
    return ReadOnlyTensorWrapper(t) if read_only else t


def cute_tensor(t, read_only=False):
    # Wrap a torch tensor as a cute tensor via the fast tvm-ffi exchange.
    ct = cute.runtime.from_dlpack(_ro(t, read_only), enable_tvm_ffi=True)
    ct.element_type = torch2cute[t.dtype]
    return ct


def cute_tensor_dynMN(t, vec, align=None, read_only=False):
    # 2D wrap with BOTH extents dynamic: rows (mode 0) fully dynamic, the
    # contiguous N (mode 1) dynamic with divisibility=vec so the kernel may keep
    # emitting vec-wide (up to 128-bit) loads. One compiled kernel then serves a
    # whole (vec-class, tile-ceiling) BUCKET of shapes instead of one exact N --
    # the K1 recompile-minimization mode. Caller guarantees t is 2D row-major
    # compact and t.shape[1] % vec == 0 (the bucket's vec class).
    w = _ro(t, read_only)
    ct = (
        cute.runtime.from_dlpack(w, assumed_align=align, enable_tvm_ffi=True)
        if align is not None
        else cute.runtime.from_dlpack(w, enable_tvm_ffi=True)
    )
    ct.element_type = torch2cute[t.dtype]
    ct.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
    ct.mark_compact_shape_dynamic(mode=1, stride_order=(0, 1), divisibility=vec)
    return ct


def cute_tensor_dynM(t, align=None, ndim=None, read_only=False):
    # Like cute_tensor but marks the LEADING dim (mode 0 = the M / output-row axis)
    # DYNAMIC while keeping the others static. For row reductions M is just the grid
    # size; baking it forces a recompile per distinct M (e.g. every batch size in a
    # training loop). mark_compact_shape_dynamic(mode=0) lets ONE compiled kernel
    # serve any M at a fixed N. stride_order is row-major outer->inner: (0,1) for 2D
    # (M, N), (0,) for 1D (M,). N stays static so the kernel's const_expr vec/tile
    # checks still resolve.
    # A single-element tensor is contiguous by definition -- with one element the stride
    # is unobservable, since every stride addresses the same element -- but torch may
    # still carry a leftover non-unit stride from the view that produced it
    # (a.diagonal(offset=2) on (5,3) gives shape (1,) stride (4,)). The DSL compares the
    # declared stride against stride_order and rejects that outright ("The stride_order
    # is not consistent with the layout"), so restride to the canonical contiguous form
    # first. Same tensor, same data, stride the DSL accepts.
    if t.numel() == 1:
        t = t.as_strided(t.shape, (1,) * t.dim())
    w = _ro(t, read_only)
    ct = (
        cute.runtime.from_dlpack(w, assumed_align=align, enable_tvm_ffi=True)
        if align is not None
        else cute.runtime.from_dlpack(w, enable_tvm_ffi=True)
    )
    ct.element_type = torch2cute[t.dtype]
    nd = ndim if ndim is not None else t.dim()
    ct.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(nd)), divisibility=1)
    return ct


# tvm-ffi launcher. Compile every native CuteDSL kernel through this (not bare
# cute.compile) for the fast arg-passing convention that skips per-arg
# get_c_pointers marshalling. The `--enable-tvm-ffi` codegen flag is equivalent to
# the cute.compile[EnableTVMFFI] typed form (measured identical host dispatch on
# B200, +-0.03us) and matches the convention used by the other native CuteDSL ops.
def compile(op, *args):
    return cute.compile(op, *args, options="--enable-tvm-ffi")


def stream():
    # Live current stream handle, read every call (never cached: callers may set a
    # different stream/device per call, and the kernel must launch on whatever is
    # current, including the CUDA-graph capture stream). _cuda_getCurrentRawStream
    # returns the real cudaStream_t pointer cheaply and capture-correctly.
    import cuda.bindings.driver as cuda

    dev = torch.cuda.current_device()
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(dev))
