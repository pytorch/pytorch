# Generic elementwise (pointwise) CuteDSL kernels.
#
# One op-agnostic kernel family serves every row of the pointwise definition table.
# Each thread applies the row's @cute.jit `fn` (with baked scalar consts) to inputs
# converted to the compute dtype, storing each output cast to its out dtype. No
# cross-thread communication.
#
# TWO paths, chosen host-side:
#   FAST (run_vec): all operands contiguous, identical shape, numel % V == 0. The
#     arrays coalesce to a flat (numel/V, V) layout; each thread vector-loads a
#     V-wide fragment (V*dtype = 128 bits -> wide global load), computes, stores.
#     This is the bandwidth path and hits ~parity with aten.
#   GENERAL (run_strided): anything else (broadcast / transposed / strided / ragged
#     numel). Operands are expanded to the broadcast shape and wrapped via
#     from_dlpack, which carries each operand's real layout (broadcast dims are
#     stride-0); the kernel indexes linearly and CuTe decodes the offset. Correct
#     for all cases, not vectorized.
#
# Addressing is canonical CuTe in both paths -- no hand-rolled offset math.

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

import torch

from .._cutedsl import launch as _L
from .._cutedsl.plan_cache import cached_plan


_BLOCK = 256
_PLAN = {}  # key -> _Plan (compiled kernel + shape-invariant launch decisions)


def _vec_width(compute_bits: int) -> int:
    # Elements per 128-bit vector for the compute dtype (fp32->4, fp16/bf16->8,
    # fp64->2). The fast path requires numel divisible by this.
    return max(128 // compute_bits, 1)


class _ElementwiseVec:
    # Vectorized flat path. Operands are (nvec, V) cute tensors; each thread owns one
    # V-wide row. fn/consts/compute/out_types as in the strided op.
    def __init__(self, fn, nin, nout, consts, compute, out_types, V):
        self.fn = fn
        self.nin = nin
        self.nout = nout
        self.consts = consts
        self.compute = compute
        self.out_types = out_types
        self.V = V

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, stream: cuda.CUstream):
        nvec = mOuts[0].shape[0]
        self.kernel(mIns, mOuts, nvec).launch(
            grid=[cute.ceil_div(nvec, _BLOCK), 1, 1],
            block=[_BLOCK, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list, nvec: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * _BLOCK + tidx
        if i < nvec:
            V = const_expr(self.V)
            # Wide vector load of each input's V-element row into registers.
            regs = []
            for k in cutlass.range_constexpr(self.nin):
                g = mIns[k][i, None]
                r = cute.make_rmem_tensor_like(g)
                cute.autovec_copy(g, r)
                regs.append(r)
            outs = [
                cute.make_rmem_tensor_like(mOuts[j][i, None]) for j in range(self.nout)
            ]
            # Per-element compute over the fragment.
            for e in cutlass.range_constexpr(V):
                vals = tuple(
                    self.compute(regs[k][e]) for k in range(const_expr(self.nin))
                )
                res = self.fn(*vals, *self.consts)
                if const_expr(self.nout == 1):
                    outs[0][e] = self.out_types[0](res)
                else:
                    for j in cutlass.range_constexpr(self.nout):
                        outs[j][e] = self.out_types[j](res[j])
            for j in cutlass.range_constexpr(self.nout):
                cute.autovec_copy(outs[j], mOuts[j][i, None])


class _ElementwiseStrided:
    # General path: one thread per output element, linear index into each operand's
    # (possibly broadcast / strided) layout -- CuTe decodes the offset.
    def __init__(self, fn, nin, nout, consts, compute, out_types):
        self.fn = fn
        self.nin = nin
        self.nout = nout
        self.consts = consts
        self.compute = compute
        self.out_types = out_types

    @cute.jit
    def __call__(self, mIns: list, mOuts: list, stream: cuda.CUstream):
        n = cute.size(mOuts[0])
        self.kernel(mIns, mOuts, n).launch(
            grid=[cute.ceil_div(n, _BLOCK), 1, 1], block=[_BLOCK, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIns: list, mOuts: list, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * _BLOCK + tidx
        if i < n:
            vals = tuple(self.compute(mIns[k][i]) for k in range(const_expr(self.nin)))
            outs = self.fn(*vals, *self.consts)
            if const_expr(self.nout == 1):
                mOuts[0][i] = self.out_types[0](outs)
            else:
                for j in cutlass.range_constexpr(self.nout):
                    mOuts[j][i] = self.out_types[j](outs[j])


def _vec_ok(inputs, shape, out_dtypes, compute_torch, V):
    # Fast path requires: every input already has the output shape (no broadcast),
    # is contiguous, 16-byte aligned (the (numel/V, V) wrap asserts
    # assumed_align=16 and from_dlpack RAISES on a misaligned base -- e.g. a
    # sliced view from diff/istft decompositions; those must take the strided
    # path), the element count is a multiple of V, AND every output dtype
    # equals the compute dtype. The last rules out bool (gt) and mixed
    # (frexp -> int32) outputs, whose narrow / non-compute element widths
    # can't take the wide vector copy. Those go general.
    if any(d != compute_torch for d in out_dtypes):
        return False
    numel = math.prod(shape)
    if numel == 0 or numel % V != 0:
        return False
    return all(
        tuple(t.shape) == tuple(shape)
        and t.is_contiguous()
        and (t.storage_offset() * t.element_size()) % 16 == 0
        for t in inputs
    )


class _Plan(NamedTuple):
    path: str  # "vec" | "strided"
    shape: tuple  # broadcast output shape
    V: int  # vector width (vec path)
    out_dtypes: tuple  # per-output torch dtype (its length is the output count)
    fn: object  # the compiled kernel


def _build_plan(fn, nin, nout, consts, compute, compute_torch, inputs, out_dtypes):
    # ALL shape-invariant work for this operand signature: broadcast shape, path
    # selection, op construction, and the kernel compile (against the live tensors as
    # seeds -- their layout matches every later call with this key). Run once per
    # key; the result is memoized so repeat calls only alloc + wrap + launch.
    shape = tuple(torch.broadcast_shapes(*(t.shape for t in inputs)))
    out_types = [_L.torch2cute[d] for d in out_dtypes]
    V = _vec_width(compute.width)
    dev = inputs[0].device
    seed_outs = [torch.empty(shape, device=dev, dtype=d) for d in out_dtypes]
    if _vec_ok(inputs, shape, out_dtypes, compute_torch, V):
        op = _ElementwiseVec(fn, nin, nout, consts, compute, out_types, V)
        cin = [_L.cute_tensor_vec(t, V, read_only=True) for t in inputs]
        cout = [_L.cute_tensor_vec(o, V) for o in seed_outs]
        path = "vec"
    else:
        op = _ElementwiseStrided(fn, nin, nout, consts, compute, out_types)
        cin = [_L.cute_tensor(t.expand(shape), read_only=True) for t in inputs]
        cout = [_L.cute_tensor(o) for o in seed_outs]
        path = "strided"
    compiled = _L.compile(op, cin, cout, _L.stream())
    return _Plan(path, shape, V, tuple(out_dtypes), compiled)


def run(fn, key, nin, nout, consts, compute, compute_torch, inputs, out_dtypes):
    # inputs: torch tensors (any broadcastable shapes / strides). Returns nout torch
    # tensors of the broadcast shape. The plan (path / shape / compiled kernel) is
    # memoized per `key`; a cache hit does only the irreducible per-call work --
    # allocate outputs, wrap the live operands, launch.
    plan = cached_plan(
        _PLAN,
        key,
        lambda: _build_plan(
            fn, nin, nout, consts, compute, compute_torch, inputs, out_dtypes
        ),
    )
    dev = inputs[0].device
    outs = [torch.empty(plan.shape, device=dev, dtype=d) for d in plan.out_dtypes]
    if plan.path == "vec":
        cin = [_L.cute_tensor_vec(t, plan.V, read_only=True) for t in inputs]
        cout = [_L.cute_tensor_vec(o, plan.V) for o in outs]
    else:
        cin = [_L.cute_tensor(t.expand(plan.shape), read_only=True) for t in inputs]
        cout = [_L.cute_tensor(o) for o in outs]
    plan.fn(cin, cout, _L.stream())
    return tuple(outs)
