# mypy: allow-untyped-defs
"""
Sorted segment-reduce lowering for low-cardinality index_put(accumulate=True).

When a large source (M rows of width D) is scatter-added into a small table (n rows),
every element becomes a global atomic. Atomics are then limited by the L2 atomic rate,
which the partitioned scatter in reduced_atomic_contention.py does not change (it only
reduces contention). Sorting the indices turns the scatter into segment sums: each program
walks a contiguous run of sorted rows with a register accumulator and issues atomics only
when the key changes. Accumulation is fp32 for fp32/bf16/fp16 inputs.

    index_put(base, [idx], src, accumulate=True)   # base [n, *T], src [M, *T]
 -> base + segment_index_add(src.view(M, D), idx, n).view(n, *T)   (rounded once)
"""

import logging

import torch
import torch.fx as fx
from torch._inductor import config
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    optimization_hint,
)
from torch.utils._triton import has_triton


log = logging.getLogger(__name__)
aten = torch.ops.aten

_MAX_ROWS = 65536
_MIN_ROW_WIDTH = 64


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _seg_combine(k1, v1, k2, v2):
        return k2, v2 + tl.where(k1 == k2, v1, 0.0)

    @triton.jit
    def _sorted_segment_sum_kernel(
        src,
        perm,
        skey,
        out,
        M,
        D,
        R: tl.constexpr,
        DB: tl.constexpr,
        NBLK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dcol = tl.program_id(1) * DB + tl.arange(0, DB)
        dmask = dcol < D
        r_ar = tl.arange(0, R)
        start = pid * NBLK * R
        # current key and its running row sum
        ckey = tl.load(skey + start)
        carry = tl.zeros([DB], dtype=tl.float32)
        for b in range(NBLK):
            rows = start + b * R + r_ar
            rm = rows < M
            k = tl.load(skey + rows, rm, other=2147483647)
            srow = tl.load(perm + rows, rm, other=0)
            v = tl.load(
                src + srow[:, None].to(tl.int64) * D + dcol[None, :],
                rm[:, None] & dmask[None, :],
                other=0.0,
            ).to(tl.float32)
            kmax = tl.max(tl.where(rm, k, -1), 0)
            kmin = tl.min(k, 0)
            if (kmin == ckey) & (kmax == ckey):
                carry += tl.sum(v, 0)
            else:
                head = rm & (k == ckey)
                carry += tl.sum(tl.where(head[:, None], v, 0.0), 0)
                tl.atomic_add(
                    out + ckey.to(tl.int64) * D + dcol,
                    carry,
                    dmask & (ckey >= 0),
                    sem="relaxed",
                )
                kk = tl.broadcast_to(k[:, None], (R, DB))
                _, vs = tl.associative_scan(
                    (kk, tl.where(head[:, None], 0.0, v)), 0, _seg_combine
                )
                knext = tl.load(skey + rows + 1, (rows + 1) < M, other=-2)
                seg_end = rm & (~head) & (k != knext) & (k != kmax)
                tl.atomic_add(
                    out + k[:, None].to(tl.int64) * D + dcol[None, :],
                    vs,
                    seg_end[:, None] & dmask[None, :],
                    sem="relaxed",
                )
                # a fully masked tail block has kmax == -1: keep the previous key
                ckey = tl.where(kmax >= 0, kmax, ckey)
                carry = tl.sum(
                    tl.where((rm & (k == kmax) & (~head))[:, None], v, 0.0), 0
                )
        tl.atomic_add(
            out + ckey.to(tl.int64) * D + dcol,
            carry,
            dmask & (ckey >= 0),
            sem="relaxed",
        )


@torch.library.custom_op("inductor::sorted_segment_index_add", mutates_args=())
def sorted_segment_index_add(
    src: torch.Tensor, index: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """``out[index[i]] += src[i]`` for ``src`` of shape [M, D]; returns fp32 [num_rows, D]."""
    M, D = src.shape
    out = torch.zeros(num_rows, D, device=src.device, dtype=torch.float32)
    if M == 0:
        return out
    # Same contract as index_put: out-of-range indices are a (device-side) error.
    torch._assert_async(((index >= -num_rows) & (index < num_rows)).all())
    index = torch.where(index < 0, index + num_rows, index).to(torch.int32)
    skey, perm = torch.sort(index)
    src = src.contiguous()
    R, NBLK = 32, 32
    DB = 128 if D >= 128 else triton.next_power_of_2(D)
    grid = (triton.cdiv(M, R * NBLK), triton.cdiv(D, DB))
    _sorted_segment_sum_kernel[grid](
        src, perm, skey, out, M, D, R, DB, NBLK, num_warps=2
    )
    return out


@sorted_segment_index_add.register_fake
def _(src, index, num_rows):
    return src.new_empty((num_rows, src.shape[1]), dtype=torch.float32)


def _hint(x):
    if isinstance(x, torch.SymInt):
        if free_unbacked_symbols(x):
            return None
        return optimization_hint(x, fallback=None)
    return int(x)


def sorted_segment_scatter_pass(graph: fx.Graph) -> int:
    """Rewrite qualifying index_put(accumulate=True) nodes; returns the number rewritten."""
    if not has_triton() or torch.are_deterministic_algorithms_enabled():
        return 0
    min_index_size = config.sorted_segment_scatter_min_index_size
    n_rewritten = 0
    for node in list(graph.nodes):
        # functional index_put only: an in-place index_put_ must keep its mutation
        if node.op != "call_function" or node.target is not aten.index_put.default:
            continue
        if len(node.args) < 4 or node.args[3] is not True:
            continue
        base, indices, src = node.args[:3]
        if len(indices) != 1 or indices[0] is None:
            continue
        idx = indices[0]
        bv, iv, sv = (getattr(a, "meta", {}).get("val") for a in (base, idx, src))
        if bv is None or iv is None or sv is None:
            continue
        if iv.dim() != 1 or bv.dim() < 2 or sv.dim() != bv.dim():
            continue
        if bv.device.type not in ("cuda", "hip") or iv.device != bv.device:
            continue
        if sv.device != bv.device or bv.dtype not in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            continue
        if iv.dtype not in (torch.int32, torch.int64):
            continue
        # static table shape (else recompiles); M may be backed-symbolic, never unbacked
        if not all(isinstance(d, int) for d in bv.shape):
            continue
        if list(bv.shape[1:]) != list(sv.shape[1:]):
            continue
        n = bv.shape[0]
        m = _hint(sv.shape[0])
        width = 1
        for d in bv.shape[1:]:
            width *= d
        if m is None or m < min_index_size or n > _MAX_ROWS or width < _MIN_ROW_WIDTH:
            continue
        with graph.inserting_before(node):
            src2 = graph.call_function(
                aten.reshape.default, (src, [sv.shape[0], width])
            )
            seg = graph.call_function(
                torch.ops.inductor.sorted_segment_index_add.default, (src2, idx, n)
            )
            seg = graph.call_function(aten.reshape.default, (seg, list(bv.shape)))
            if bv.dtype == torch.float32:
                new = graph.call_function(aten.add.Tensor, (base, seg))
            else:
                b32 = graph.call_function(
                    torch.ops.prims.convert_element_type.default, (base, torch.float32)
                )
                new = graph.call_function(aten.add.Tensor, (b32, seg))
                new = graph.call_function(
                    torch.ops.prims.convert_element_type.default, (new, bv.dtype)
                )
        new.meta.update(node.meta)
        node.replace_all_uses_with(new)
        graph.erase_node(node)
        n_rewritten += 1
    if n_rewritten:
        graph.eliminate_dead_code()
        log.debug("sorted_segment_scatter: rewrote %d index_put nodes", n_rewritten)
    return n_rewritten


def sorted_segment_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    sorted_segment_scatter_pass(graph)
    return graph
