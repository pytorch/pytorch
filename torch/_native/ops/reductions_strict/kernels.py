"""CuTeDSL strict-numerics reduction kernels (bitwise-matches Inductor's INNER_TREE order).

Lazy: ``import``ing this module does NOT import cutlass -- the DSL + kernels are built on
the first strict call (``_dsl``), so ``import torch`` stays cutlass-free. Registered as an
aten override via ``strict_sum_impl.py`` using the torch._native framework (same pattern as
ops/norm's RMSNorm). The R0_BLOCK tile and split factor come from ``torch._strict_config``,
the SAME functions Inductor uses -- so eager and torch.compile match by construction.
"""

import functools
import math
import types

import torch

from torch._strict_config import strict_rblock, strict_split_factor  # shared with Inductor


WARP = 32

# dtypes the strict kernel supports (plain torch dtypes -> usable in `cond` WITHOUT cutlass)
SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
# accumulate (and store split partials) in fp32 for fp16/bf16/fp32, fp64 for fp64.
_ACC_TORCH = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


def _round_warp(p):
    return ((p + WARP - 1) // WARP) * WARP


def _offsets_up(tpr):
    n, offs, o = min(tpr, WARP), [], 1
    while o < n:
        offs.append(o)
        o *= 2
    return offs


@functools.cache
def _num_sm(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _dsl():
    """Import CuTeDSL + build the kernels ON FIRST STRICT CALL (never at torch import).
    Returns a namespace of the pieces strict_reduce needs. Cached, so built once."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import const_expr

    torch2cute = {
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }
    acc_ty = {
        torch.float16: cutlass.Float32,
        torch.bfloat16: cutlass.Float32,
        torch.float32: cutlass.Float32,
        torch.float64: cutlass.Float64,
    }

    class Sum:  # op = (init, reduce(acc,val,valid), combine(a,b)); add a class to extend.
        init = 0.0

        @staticmethod
        @cute.jit
        def reduce(acc, val, valid):
            return (acc + val) if valid else acc

        @staticmethod
        @cute.jit
        def combine(a, b):
            return a + b

    class StrictRowReduce:
        """Contiguous (M, N) -> (M,), reducing the last axis in strict INNER_TREE order.
        One output row per block: the GPU already packs multiple such blocks per SM (measured
        higher bandwidth than packing rows into a CTA), so 1-row blocks are optimal here."""

        def __init__(self, N, P, trait=Sum, acc=None):
            self.N = int(N)
            self.P = int(P)
            self.trait = trait
            self.acc_ty = acc or cutlass.Float32
            self.num_threads = _round_warp(self.P)
            self.warps = self.num_threads // WARP

        @cute.jit
        def __call__(self, mX: cute.Tensor, mOut: cute.Tensor, stream):
            self.kernel(mX, mOut).launch(
                grid=[mX.shape[0], 1, 1], block=[self.num_threads, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mOut: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            row, _, _ = cute.arch.block_idx()
            P = const_expr(self.P)
            N = const_expr(self.N)
            reduce = self.trait.reduce
            combine = self.trait.combine
            at = self.acc_ty
            acc = at(self.trait.init)

            if const_expr(P >= WARP and N % P == 0):        # Stage A (MAP): stride-P fold
                r = tidx
                for _ in cutlass.range(const_expr(N // P)):
                    acc = reduce(acc, at(mX[row, r]), True)
                    r = r + P
            else:
                for k in cutlass.range(const_expr((N + P - 1) // P)):  # runtime loop (not unrolled)
                    c = tidx + k * P
                    valid = (tidx < P) and (c < N)
                    cc = c if valid else 0
                    acc = reduce(acc, at(mX[row, cc]), valid)

            for off in _offsets_up(P):                       # Stage B (TREE): count-up warp
                acc = combine(acc, cute.arch.shuffle_sync_bfly(acc, offset=off))

            if const_expr(self.warps > 1):                   # cross-warp count-up via smem
                smem = cutlass.utils.SmemAllocator()
                buf = smem.allocate_tensor(
                    at, cute.make_layout(const_expr(self.warps)), byte_alignment=8
                )
                lane = cute.arch.lane_idx()
                warp = cute.arch.warp_idx()
                if lane == 0:
                    buf[warp] = acc
                cute.arch.barrier()
                v = at(self.trait.init)
                if lane < const_expr(self.warps):
                    v = buf[lane]
                for off in _offsets_up(const_expr(self.warps)):
                    v = combine(v, cute.arch.shuffle_sync_bfly(v, offset=off))
                acc = v

            if tidx == 0:
                mOut[row] = mOut.element_type(acc)

    def cute_tensor(t):
        ct = cute.runtime.from_dlpack(t, enable_tvm_ffi=True)
        ct.element_type = torch2cute[t.dtype]
        return ct

    def stream():
        dev = torch.cuda.current_device()
        return cuda.CUstream(torch._C._cuda_getCurrentRawStream(dev))

    compile_cache: dict = {}

    def launch(op, mX, mOut, in_stride, out_stride):
        # strides MUST be in the key: cute.compile bakes the input layout, and the
        # strided-direct path passes non-contiguous views -- a stride-agnostic key would
        # reuse a kernel compiled for a different layout (crash / potential miscompute).
        key = (op.N, op.P, op.trait.__name__, op.acc_ty, mX.shape[0],
               mX.element_type, mOut.element_type, tuple(in_stride), tuple(out_stride))
        fn = compile_cache.get(key)
        st = stream()
        if fn is None:
            fn = cute.compile(op, mX, mOut, st, options="--enable-tvm-ffi")
            compile_cache[key] = fn
        fn(mX, mOut, st)

    def row_reduce_2d(x2d, trait, acc, out_dtype):
        # x2d may be strided (a permuted view): the kernel indexes mX[row, r] via cute
        # strides, so we do NOT force a contiguous copy here -- that's the whole point of
        # the strided-direct path (avoids the transpose materialize). The reduction order
        # is defined explicitly in the kernel (lane t -> logical {t, t+P, ...}), so it's
        # layout-independent and still bitwise-matches the materialized/Inductor result.
        M, N = x2d.shape
        out = torch.empty((M,), device=x2d.device, dtype=out_dtype)
        op = StrictRowReduce(N, strict_rblock(N), trait, acc)
        launch(op, cute_tensor(x2d), cute_tensor(out), x2d.stride(), out.stride())
        return out

    return types.SimpleNamespace(Sum=Sum, acc_ty=acc_ty, row_reduce_2d=row_reduce_2d)


def _normalize(dim, nd):
    if dim is None or (isinstance(dim, (tuple, list)) and len(dim) == 0):
        return list(range(nd))
    if isinstance(dim, int):
        return [dim % nd]
    return sorted(d % nd for d in dim)


def strict_reduce(x, dim, keepdim=False):
    """Strict reduction over `dim` (lazily builds CuTeDSL). Called only from the override
    impl, which the framework gates on runtime availability + eligibility."""
    d = _dsl()
    trait = d.Sum
    x = x.contiguous()
    nd = x.dim()
    red = _normalize(dim, nd)
    kept = [i for i in range(nd) if i not in red]
    kept_shape = [x.shape[i] for i in kept]
    M = math.prod(kept_shape) if kept_shape else 1
    N = math.prod([x.shape[i] for i in red]) if red else 1

    acc = d.acc_ty[x.dtype]
    acc_torch = _ACC_TORCH[x.dtype]             # split partials in acc dtype
    C = strict_split_factor(N, M, _num_sm(x.device))
    if C > 1:                                                # two-stage split
        xp = x.permute(kept + red).reshape(M, N).contiguous()  # split needs contiguous chunks
        s1 = d.row_reduce_2d(xp.reshape(M * C, N // C), trait, acc, acc_torch)
        out = d.row_reduce_2d(s1.reshape(M, C), trait, acc, x.dtype)
    else:
        perm = x.permute(kept + red)            # view
        if perm.dim() == 2 and tuple(perm.shape) == (M, N):
            xp = perm                           # strided-direct: reduce the view, NO copy
        elif kept + red == list(range(nd)):
            xp = x.reshape(M, N)                # contiguous (reduced dims already innermost)
        else:
            xp = perm.reshape(M, N).contiguous()  # higher-D dim merge -> must materialize
        out = d.row_reduce_2d(xp, trait, acc, x.dtype)

    if keepdim:
        return out.reshape([1 if i in red else s for i, s in enumerate(x.shape)])
    return out.reshape(kept_shape) if kept_shape else out.reshape(())


def strict_sum(x, dim=None, keepdim=False):
    return strict_reduce(x, dim, keepdim)
