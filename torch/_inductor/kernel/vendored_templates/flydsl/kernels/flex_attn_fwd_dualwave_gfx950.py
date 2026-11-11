# mypy: allow-untyped-defs
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Dual-wave software-pipelined gfx950 forward for the DUALWAVE_HEAD_DIMS.

512-thread CTAs own 256 query rows; two 4-wave groups run one pipeline cluster
apart over all of Sk, the top-left causal tiles, or the BlockMask-listed blocks
of the CTA's two Q blocks. Q/K/V/O are BHSD views with unit-stride head dims.
"""

import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith, fly, llvm, vector
from flydsl.expr import arith, const_expr, gpu, math as fmath, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw, ArithValue
from flydsl.runtime.device import get_rocm_arch

from .flex_attn_fwd_gfx950 import _make_forward_launch
from .flex_attn_plan import dualwave_plan
from .flex_attn_target import FLEX_ARCH, resolve_target, WAVE_SIZE
from .flex_attn_utils import (
    capture_extent_count,
    capture_table_layout,
    ceil_div,
    make_capture_buffers,
    make_global_view,
    make_mask_evaluator,
    make_score_mod_readers,
    mask_capture_sizes,
    optional_int,
    resolve_capture_extents,
    runtime_extent,
    runtime_strides,
    tied_nop,
    tile_axes,
    tile_grid,
    validate_score_mod_args,
)


_LOG2E = host_math.log2(host_math.e)
# Fast-math without nnan/ninf: masked scores are -inf and must stay -inf.
_FASTMATH = (
    fx.arith.FastMathFlags.reassoc
    | fx.arith.FastMathFlags.nsz
    | fx.arith.FastMathFlags.arcp
    | fx.arith.FastMathFlags.contract
    | fx.arith.FastMathFlags.afn
)
# s_waitcnt bitfield encoding
_VMCNT_LO_MASK = 0xF
_LGKMCNT_EXPCNT_BASE = 0x3F70
_VMCNT_HI_SHIFT = 14
_VMCNT_HI_MASK = 0x3
_LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "flydsl.dualwave_swp.lds">'
_MAX_BUFFER_BYTES = 1 << 32

DUALWAVE_HEAD_DIM = 128
# (QK, V) head dims. K/V tiles are staged in 64-column LDS lines, so a head
# dim that is not a multiple of 64 stages the columns up to the next one.
DUALWAVE_HEAD_DIMS = ((64, 128), (96, 96), (128, 128))
# (QK, V) head dims of the 4-wave variant: 128-row CTAs with one wave per SIMD,
# which can address 512 registers per lane.
DUALWAVE_WIDE_HEAD_DIMS = (
    (64, 256),
    (128, 256),
    (160, 160),
    (192, 128),
    (192, 192),
    (224, 224),
    (256, 128),
    (256, 256),
)
DUALWAVE_WIDE_WAVES = 4
DUALWAVE_BLOCK_M = 256
# The pipeline needs at least six 64-row KV tiles.
DUALWAVE_MIN_SEQ_LEN = 384
# The per-CTA union of BlockMask blocks takes an int32 of LDS per KV block next
# to the K/V buffers, and two more inside them before the first K/V load.
DUALWAVE_MAX_KV_BLOCKS = 8192
# Key offsets of a lane's 16 scores in each 32-key half of a 64-key tile,
# past 4 * (lane // 32).
_TILE_KEY_OFFSETS = (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27)
# BLOCK_STATE_* of a union entry: 2 bits per Q block above the KV block index.
_STATE_SHIFT = 24
_BLOCK_INDEX_MASK = (1 << _STATE_SHIFT) - 1
_STATE_PARTIAL = 1
_STATE_FULL = 2


# K/V tiles of 64 keys are staged in LDS lines of 8 key slots by 64 columns:
# line l of a column band holds keys 8 * slot + l.
_KV_LINES = 8
_KV_LINE_COLUMNS = 64
_K_LINE_STRIDE = 520
_V_LINE_STRIDE = 544


def kv_dma_issues(num_waves, column_bands):
    """(line, key, column) of a wave's K/V DMAs per 64-key tile, past line ``wave``."""
    issues = _KV_LINES // num_waves
    return tuple(
        (
            issue * num_waves + band * _KV_LINES,
            issue * num_waves,
            band * _KV_LINE_COLUMNS,
        )
        for band in range(column_bands)
        for issue in range(issues)
    )


def k_read_lane_offset(lane_mod_32, lane_div_32):
    """LDS element of a lane's first K pack in a K tile: key ``lane % 32``."""
    return (
        (lane_mod_32 % _KV_LINES) * _K_LINE_STRIDE
        + (lane_mod_32 // _KV_LINES) * _KV_LINE_COLUMNS
        + lane_div_32 * 8
    )


def k_read_step_offset(k_step):
    """Offset of a lane's K pack for columns ``16 * k_step`` on; keys 32-63 add 256."""
    return (k_step // 4) * (_KV_LINES * _K_LINE_STRIDE) + (k_step % 4) * 16


def dualwave_kv_lds_bytes(head_dim, v_head_dim):
    """LDS bytes of the double-buffered 2-byte K and V tiles."""
    k_bands, v_bands = (
        ceil_div(dim, _KV_LINE_COLUMNS) for dim in (head_dim, v_head_dim)
    )
    return 2 * 2 * _KV_LINES * (k_bands * _K_LINE_STRIDE + v_bands * _V_LINE_STRIDE)


def dualwave_max_kv_blocks(head_dim, v_head_dim, target=None):
    """128-key blocks a block-list build can take next to its K/V tiles in
    ``target``'s LDS (see ``resolve_target``)."""
    free = resolve_target(target).lds_bytes - dualwave_kv_lds_bytes(
        head_dim, v_head_dim
    )
    return min(DUALWAVE_MAX_KV_BLOCKS, free // 4 - 4)


def _dtype_to_elem_type(dtype_str):
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "f16":
        return fx.Float16
    raise ValueError(f"unsupported dtype: {dtype_str!r}")


def _ds_read_tr16_b64_imm(
    result_type, lds_ptr, imm_offset, alias_scopes, noalias_scopes
):
    # Inline asm here would get no lgkmcnt wait before uses of its result.
    ptr = llvm.GEPOp(
        lds_ptr.type, lds_ptr, [], [int(imm_offset)], T.i8, llvm.GEPNoWrapFlags.inbounds
    ).result
    raw = rocdl.ds_read_tr16_b64(
        ir.VectorType.get([4], ir.F16Type.get()),
        ptr,
        alias_scopes=alias_scopes,
        noalias_scopes=noalias_scopes,
    ).result
    return vector.BitCastOp(result_type, raw).result


def _lds_ptr_from_index(value):
    value = _raw(value)
    if isinstance(value.type, ir.IndexType):
        value = _std_arith.IndexCastOp(T.i64, value).result
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), value).result


def _lds_byte_gep(base_ptr, byte_offset):
    offset = _raw(byte_offset)
    if isinstance(offset.type, ir.IndexType):
        offset = _std_arith.IndexCastOp(T.i64, offset).result
    return llvm.GEPOp(base_ptr.type, base_ptr, [offset], [-(2**31)], T.i8, None).result


def _lds_i32_ptr(base_ptr, index):
    return _lds_byte_gep(base_ptr, fx.Int32(index) * fx.Int32(4))


def _lds_store_i32(base_ptr, index, value):
    llvm.StoreOp(_raw(fx.Int32(value)), _lds_i32_ptr(base_ptr, index))


def _lds_load_i32(base_ptr, index):
    return fx.Int32(llvm.LoadOp(T.i32, _lds_i32_ptr(base_ptr, index)).result)


@flyc.jit
def build_block_union(
    lists,
    first_row,
    second_valid,
    kv_blocks,
    state_ptr,
    union_ptr,
    kv_block_cap,
    num_threads,
):
    """Write the ascending union of the KV blocks BlockMask rows ``first_row``
    and (if ``second_valid``; never if None) ``first_row + 1`` list to
    ``union_ptr``; return its count, padded to at least 2. Every thread of the
    CTA must call this.

    ``lists`` is (counts, indices, row width) of the partial and full lists;
    entry bits 24-25 / 26-27 hold each row's BLOCK_STATE_*. ``state_ptr`` takes
    ``2 * kv_block_cap + 1`` int32 of LDS, ``union_ptr`` ``kv_block_cap + 4``
    with the count at ``kv_block_cap + 2``.
    """
    state_dump = 2 * kv_block_cap
    union_dump = kv_block_cap + 3
    thread = fx.Int32(gpu.thread_idx.x)
    for step in range_constexpr(ceil_div(state_dump + 1, num_threads)):
        slot = thread + fx.Int32(step * num_threads)
        if const_expr((step + 1) * num_threads > state_dump + 1):
            slot = (slot <= fx.Int32(state_dump)).select(slot, fx.Int32(state_dump))
        _lds_store_i32(state_ptr, slot, 0)
    fx.gpu.barrier()
    # Full after partial: a block on both lists is full.
    for list_index in range_constexpr(2):
        counts_view, indices_view, width = lists[list_index]
        state = _STATE_PARTIAL if list_index == 0 else _STATE_FULL
        for row_index in range_constexpr(1 if second_valid is None else 2):
            row = first_row + fx.Int32(row_index)
            if const_expr(row_index == 1):
                row = second_valid.select(row, first_row)
            listed = fx.Int32(
                rocdl.readfirstlane(T.i32, _raw(fx.Int32(counts_view[row])))
            )
            if const_expr(row_index == 1):
                listed = second_valid.select(listed, fx.Int32(0))
            for first_slot in range(fx.Int32(0), listed, fx.Int32(num_threads)):
                list_slot = first_slot + thread
                in_list = list_slot < listed
                listed_block = fx.Int32(
                    indices_view[row * width + in_list.select(list_slot, fx.Int32(0))]
                )
                kept = (
                    in_list & (listed_block >= fx.Int32(0)) & (listed_block < kv_blocks)
                )
                _lds_store_i32(
                    state_ptr,
                    kept.select(
                        listed_block + fx.Int32(row_index * kv_block_cap),
                        fx.Int32(state_dump),
                    ),
                    state,
                )
        fx.gpu.barrier()

    wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(thread // fx.Int32(WAVE_SIZE))))
    if wave == fx.Int32(0):
        lane = thread % fx.Int32(WAVE_SIZE)
        found = fx.Int32(0)
        for first_block in range(fx.Int32(0), kv_blocks, fx.Int32(WAVE_SIZE)):
            block = first_block + lane
            in_range = block < kv_blocks
            safe_block = in_range.select(block, fx.Int32(0))
            state_a = _lds_load_i32(state_ptr, safe_block)
            state_b = _lds_load_i32(state_ptr, safe_block + fx.Int32(kv_block_cap))
            visited = in_range & ((state_a | state_b) != fx.Int32(0))
            visited_lanes = rocdl.ballot(T.i64, _raw(visited))
            below = rocdl.mbcnt_hi(
                T.i32,
                _std_arith.TruncIOp(
                    T.i32, _raw(fx.Int64(visited_lanes) >> fx.Int64(32))
                ).result,
                rocdl.mbcnt_lo(
                    T.i32,
                    _std_arith.TruncIOp(T.i32, visited_lanes).result,
                    _raw(fx.Int32(0)),
                ),
            )
            _lds_store_i32(
                union_ptr,
                visited.select(found + fx.Int32(below), fx.Int32(union_dump)),
                block
                | (state_a << fx.Int32(_STATE_SHIFT))
                | (state_b << fx.Int32(_STATE_SHIFT + 2)),
            )
            found = found + fx.Int32(fmath.ctpop(fx.Int64(visited_lanes)))
        # The pipeline takes two blocks: pad with unlisted blocks that end
        # below Sk, keeping the order ascending.
        first_entry = _lds_load_i32(union_ptr, 0)
        second_entry = _lds_load_i32(union_ptr, 1)
        first_listed = first_entry & fx.Int32(_BLOCK_INDEX_MASK)
        none = found == fx.Int32(0)
        one = found == fx.Int32(1)
        shift_up = one & (first_listed != fx.Int32(0))
        _lds_store_i32(
            union_ptr,
            0,
            none.select(
                fx.Int32(0), shift_up.select(first_listed - fx.Int32(1), first_entry)
            ),
        )
        _lds_store_i32(
            union_ptr,
            1,
            none.select(
                fx.Int32(1),
                one.select(shift_up.select(first_entry, fx.Int32(1)), second_entry),
            ),
        )
        _lds_store_i32(
            union_ptr,
            kv_block_cap + 2,
            (found < fx.Int32(2)).select(fx.Int32(2), found),
        )
    fx.gpu.barrier()
    return fx.Int32(
        rocdl.readfirstlane(T.i32, _raw(_lds_load_i32(union_ptr, kv_block_cap + 2)))
    )


def _concat_pairwise(packs, concat):
    """Concatenate ``packs`` in order as a balanced tree of ``concat``."""
    while len(packs) > 1:
        pairs = [concat(packs[i], packs[i + 1]) for i in range(0, len(packs) - 1, 2)]
        packs = pairs + packs[len(pairs) * 2 :]
    return packs[0]


def _waitcnt_vm_n(n):
    """Emit s_waitcnt vmcnt(n) only (lgkmcnt=63, expcnt=7)."""
    val = (
        (n & _VMCNT_LO_MASK)
        | _LGKMCNT_EXPCNT_BASE
        | (((n >> 4) & _VMCNT_HI_MASK) << _VMCNT_HI_SHIFT)
    )
    rocdl.s_waitcnt(val)


def _read_exec_i64():
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _lds_alias_scope_array(names):
    attrs = [
        f'#llvm.alias_scope<id = "{name}", domain = {_LDS_ALIAS_DOMAIN}>'
        for name in names
    ]
    return ir.Attribute.parse(f"[{', '.join(attrs)}]")


def flat_buffer_view(tensor, num_records_bytes=None, offset=None):
    """``tensor``'s memory as a buffer view whose ``(None, i)`` slice is memory
    element ``offset + i``, not coordinate i of ``tensor``'s layout."""
    size = fx.get_scalar(fx.cosize(fx.get_layout(tensor)))
    iterator = fx.get_iter(tensor)
    if offset is not None:
        iterator = iterator + offset
    view = fx.make_view(iterator, fx.make_layout(size, 1))
    return fx.logical_divide(
        fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes),
        fx.make_layout(1, 1),
    )


def tile_buffer_view(tensor, end_elems, tile_elems, element_bytes=2):
    """``flat_buffer_view`` from element ``tile_elems`` whose loads read zeros
    from element ``end_elems`` on, also for a tile that starts past it."""
    # A negative byte count would wrap to an unbounded descriptor.
    records = fx.Int64(end_elems) - fx.Int64(tile_elems)
    records = (records > fx.Int64(0)).select(records, fx.Int64(0))
    return flat_buffer_view(
        tensor, records * fx.Int64(element_bytes), fx.Int32(tile_elems)
    )


def bshd_strides(seq_len, num_heads, head_dim=DUALWAVE_HEAD_DIM):
    """BHSD-view strides of a contiguous [B, S, H, D] tensor; a None ``seq_len``
    gives a runtime batch stride."""
    batch_stride = None if seq_len is None else seq_len * num_heads * head_dim
    return (batch_stride, head_dim, num_heads * head_dim, 1)


def build_flex_attn_fwd_dualwave_module(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    kv_seq_len: int | None = None,
    head_dim: int = DUALWAVE_HEAD_DIM,
    v_head_dim: int | None = None,
    scale: float,
    causal: bool,
    dtype_str: str = "bf16",
    q_stride=None,
    k_stride=None,
    v_stride=None,
    o_stride=None,
    kv_head_dims=None,
    write_max_scores: bool = True,
    plan=None,
    num_waves: int = 8,
    waves_per_eu: int | None = None,
    score_mod=None,
    score_mod_aux=(),
    score_mod_buffer_count: int = 0,
    score_mod_buffer_dtypes=(),
    mod_key=None,
    block_lists: bool = False,
    block_mask_batch=1,
    block_mask_heads: int = 1,
    num_q_blocks=None,
    max_partial_blocks=None,
    max_full_blocks=None,
    mask_program=(),
    mask_program_output: int = 0,
    mask_buffer_shapes=(),
    mask_buffer_strides=(),
    mask_buffer_dtypes=(),
    head_slow_order: bool = False,
    target_limits=None,
):
    """Return a JIT launcher taking (query, key, value, logsumexp, max_scores, output, stream).

    None sizes are read at runtime; a runtime Sk needs ``plan``. With a
    ``score_mod`` or ``block_lists`` the launcher takes the BlockMask metadata
    and capture slots of ``build_flex_attn_fwd_module`` before the output.
    ``head_slow_order`` launches the grid in ``tile_axes`` head-slow order.
    ``num_waves=4`` builds the 128-row variant for ``DUALWAVE_WIDE_HEAD_DIMS``,
    where ``kv_head_dims`` may give narrower stored K/V widths: Q and O rows
    are then zero-padded to ``head_dim`` and ``v_head_dim``, and the attention
    runs at the K/V widths. ``target_limits`` bound the block-list LDS, by
    default the current device's.
    """

    batch_size = optional_int(batch_size)
    num_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    seq_len = optional_int(seq_len)
    kv_seq_len = seq_len if kv_seq_len is None else optional_int(kv_seq_len)
    head_dim = int(head_dim)
    v_head_dim = head_dim if v_head_dim is None else int(v_head_dim)
    NUM_WAVES = int(num_waves)
    q_row_dim, o_row_dim = head_dim, v_head_dim
    if kv_head_dims is not None:
        head_dim, v_head_dim = (int(dim) for dim in kv_head_dims)
        if NUM_WAVES != DUALWAVE_WIDE_WAVES or not (
            head_dim <= q_row_dim and v_head_dim <= o_row_dim
        ):
            raise ValueError(
                "dual-wave forward takes kv_head_dims up to the head dims, "
                "with 4 waves only"
            )
    if NUM_WAVES == 8:
        if (head_dim, v_head_dim) not in DUALWAVE_HEAD_DIMS:
            raise ValueError(
                f"dual-wave forward supports (QK, V) head dims {DUALWAVE_HEAD_DIMS}, "
                f"got ({head_dim}, {v_head_dim})"
            )
    elif NUM_WAVES == DUALWAVE_WIDE_WAVES:
        if (head_dim, v_head_dim) not in DUALWAVE_WIDE_HEAD_DIMS:
            raise ValueError(
                "4-wave dual-wave forward supports (QK, V) head dims "
                f"{DUALWAVE_WIDE_HEAD_DIMS}, got ({head_dim}, {v_head_dim})"
            )
    else:
        raise ValueError(f"dual-wave forward runs 8 or 4 waves, got {NUM_WAVES}")
    if waves_per_eu is None:
        waves_per_eu = 2 if NUM_WAVES == 8 else 1
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dual-wave forward supports bf16/f16 only, got {dtype_str}")
    if (batch_size is not None and batch_size <= 0) or (
        num_kv_heads <= 0 or num_heads % num_kv_heads
    ):
        raise ValueError("dual-wave forward requires B > 0 and Hq % Hkv == 0")
    if seq_len is not None and seq_len <= 0:
        raise ValueError("dual-wave forward requires Sq > 0")
    if kv_seq_len is not None and kv_seq_len < DUALWAVE_MIN_SEQ_LEN:
        raise ValueError(f"dual-wave forward requires Sk >= {DUALWAVE_MIN_SEQ_LEN}")
    if kv_seq_len is None and plan is None:
        raise ValueError("dual-wave forward needs a dualwave_plan for a runtime Sk")
    strides = []
    for name, stride, seq, heads, dim in (
        ("query", q_stride, seq_len, num_heads, q_row_dim),
        ("key", k_stride, kv_seq_len, num_kv_heads, head_dim),
        ("value", v_stride, kv_seq_len, num_kv_heads, v_head_dim),
        ("output", o_stride, seq_len, num_heads, o_row_dim),
    ):
        stride = bshd_strides(seq, heads, dim) if stride is None else tuple(stride)
        # 16-byte DMA and vector accesses start at every (b, h, s) row.
        if (
            len(stride) != 4
            or stride[3] != 1
            or any(
                isinstance(entry, int)
                and entry % 8
                or isinstance(entry, tuple)
                and entry[1] % 8
                for entry in stride[:3]
            )
        ):
            raise ValueError(
                f"dual-wave forward requires {name} strides (B, H, S, 1) in "
                f"multiples of 8 elements, got {stride}"
            )
        strides.append(stride)
    q_stride, k_stride, v_stride, o_stride = strides
    # Zero K and V batch strides read one K/V batch for every query batch.
    kv_broadcast = batch_size != 1 and k_stride[0] == 0 and v_stride[0] == 0
    for stride, seq, heads, dim in (
        (q_stride, seq_len, num_heads, q_row_dim),
        (k_stride, kv_seq_len, num_kv_heads, head_dim),
        (v_stride, kv_seq_len, num_kv_heads, v_head_dim),
        (o_stride, seq_len, num_heads, o_row_dim),
    ):
        extents = (batch_size, heads, seq, dim)
        if (
            all(isinstance(value, int) for value in (*extents, *stride))
            and (1 + sum((size - 1) * step for size, step in zip(extents, stride))) * 2
            >= _MAX_BUFFER_BYTES
        ):
            raise ValueError(
                "dual-wave forward requires every tensor to be smaller than 4 GiB"
            )
    has_score_mod = score_mod is not None
    score_mod_aux, score_mod_buffer_count = validate_score_mod_args(
        score_mod, score_mod_aux, score_mod_buffer_count, mod_key
    )
    BLOCK_LISTS = bool(block_lists)
    mask_program = tuple(mask_program)
    mask_program_output = int(mask_program_output)
    mask_buffer_shapes = tuple(tuple(shape) for shape in mask_buffer_shapes)
    mask_buffer_strides = tuple(tuple(stride) for stride in mask_buffer_strides)
    mask_buffer_count = len(mask_buffer_shapes)
    if len(mask_buffer_strides) != mask_buffer_count:
        raise ValueError("mask buffer shape/stride descriptors must match")
    if (mask_program or mask_buffer_count) and not BLOCK_LISTS:
        raise ValueError("dual-wave forward evaluates mask programs on block lists")
    if BLOCK_LISTS and causal:
        raise ValueError("dual-wave forward takes causal or block lists, not both")
    if block_mask_batch not in (1, batch_size, None) or block_mask_heads not in (
        1,
        num_kv_heads,
        num_heads,
    ):
        raise ValueError("BlockMask batch must be 1 or B and heads 1, Hkv or Hq")
    # Mods take the launch signature of build_flex_attn_fwd_module.
    HAS_MODS = has_score_mod or BLOCK_LISTS
    mask_buffer_sizes = mask_capture_sizes(mask_buffer_shapes, mask_buffer_strides)
    capture_table = capture_table_layout(
        mask_buffer_sizes,
        tuple(mask_buffer_dtypes),
        score_mod_aux,
        score_mod_buffer_count,
        tuple(score_mod_buffer_dtypes),
    )
    capture_slots = capture_extent_count(mask_buffer_sizes, capture_table)
    if BLOCK_LISTS:
        if kv_seq_len is not None:
            KV_BLOCK_CAP = ceil_div(kv_seq_len, 128)
        elif plan is not None and "kv_blocks" in plan:
            KV_BLOCK_CAP = int(plan["kv_blocks"])
        else:
            raise ValueError("dual-wave block lists need a kv_blocks plan bound")
        max_kv_blocks = dualwave_max_kv_blocks(head_dim, v_head_dim, target_limits)
        if KV_BLOCK_CAP > max_kv_blocks:
            raise ValueError(
                f"dual-wave block lists take at most {max_kv_blocks} KV blocks"
            )

    scale_log2e = float(scale) * _LOG2E
    WRITE_MAX_SCORES = bool(write_max_scores)
    SEQ_LEN = seq_len
    KV_SEQ_LEN = kv_seq_len
    ROWS_PER_WAVE = 32
    BLOCK_M = ROWS_PER_WAVE * NUM_WAVES
    # Waves 4-7 run one pipeline cluster behind their SIMD's partner in 0-3.
    STAGGER = NUM_WAVES == 8
    BLOCK_N = 64
    if plan is None:
        plan = dualwave_plan(KV_SEQ_LEN)
    MASK_LAST_TILE = plan["mask_last_tile"]
    MASK_SECOND_LAST_TILE = plan["mask_second_last_tile"]
    CAUSAL = bool(causal)
    HEAD_SLOW_ORDER = bool(head_slow_order)
    HEAD_AXIS, BLOCK_AXIS = tile_axes(HEAD_SLOW_ORDER)
    # With a runtime Sk, the causal peel, taken only by q-blocks that reach Sk,
    # spills the loop state around its branch; every iteration bounds its
    # last K load instead, as with block lists, whose last block may be anywhere.
    PEEL_LAST_ITERATION = (
        MASK_SECOND_LAST_TILE
        and not (CAUSAL and KV_SEQ_LEN is None)
        and not block_lists
    )
    BOUND_LOOP_K = MASK_SECOND_LAST_TILE and not PEEL_LAST_ITERATION
    K_SUB_N = 32
    WARP_SIZE = WAVE_SIZE
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE

    HEAD_DIM = head_dim
    V_HEAD_DIM = v_head_dim
    K_STEP_QK = 16
    K_STEPS_QK = HEAD_DIM // K_STEP_QK
    D_CHUNK = 32
    D_CHUNKS = V_HEAD_DIM // D_CHUNK
    PV_K_STEP = 16
    PV_K_STEPS = K_SUB_N // PV_K_STEP
    MFMA_LANE_K = 8

    NUM_HEADS_Q = num_heads
    NUM_HEADS_KV = num_kv_heads
    GQA_GROUP_SIZE = NUM_HEADS_Q // NUM_HEADS_KV

    # LDS: interleaved double buffer K0,V0,K1,V1 (68096 B at D128); K/V lines
    # of 8 rows x 64 columns are padded to 520/544 elements.
    ELEM_BYTES = 2
    D_128B_SIZE = 64
    VEC_KV = 8
    SMEM_LINEAR_WAVE = WARP_SIZE * 16 // ELEM_BYTES
    SMEM_N_PER_WAVE = SMEM_LINEAR_WAVE // D_128B_SIZE
    SMEM_N_RPT = BLOCK_N // SMEM_N_PER_WAVE
    SMEM_K_D_RPT = ceil_div(HEAD_DIM, D_128B_SIZE)
    SMEM_V_D_RPT = ceil_div(V_HEAD_DIM, D_128B_SIZE)
    SMEM_K_PAD = 16 // ELEM_BYTES
    SMEM_V_PAD = 64 // ELEM_BYTES
    SMEM_K_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_K_PAD
    SMEM_V_LINE_STRIDE = SMEM_LINEAR_WAVE + SMEM_V_PAD
    SMEM_K_TILE_ELEMS = SMEM_N_RPT * SMEM_K_D_RPT * SMEM_K_LINE_STRIDE
    SMEM_V_TILE_ELEMS = SMEM_N_RPT * SMEM_V_D_RPT * SMEM_V_LINE_STRIDE
    NUM_PREFETCH_K = 2
    DUALWAVE_SWP_KV_PER_BUFFER = SMEM_K_TILE_ELEMS + SMEM_V_TILE_ELEMS
    LDS_KV_TOTAL_SIZE = NUM_PREFETCH_K * DUALWAVE_SWP_KV_PER_BUFFER
    DUALWAVE_SWP_K_BUF_BASE = (0, DUALWAVE_SWP_KV_PER_BUFFER)
    DUALWAVE_SWP_V_BUF_BASE = (
        SMEM_K_TILE_ELEMS,
        SMEM_K_TILE_ELEMS + DUALWAVE_SWP_KV_PER_BUFFER,
    )
    DUALWAVE_SWP_URK_N_STRIP_STRIDE = 256
    # V ds_read_tr lane base: (lane/32)*GRPK + ((lane%16)/4)*LANE_HI
    #   + ((lane/16)%2)*GRP_N + (lane%4)*LANE_LO.
    DUALWAVE_SWP_URV_GRPK = 2176
    DUALWAVE_SWP_URV_LANE_HI = SMEM_V_LINE_STRIDE
    DUALWAVE_SWP_URV_GRP_N = 16
    DUALWAVE_SWP_URV_LANE_LO = 4
    DUALWAVE_SWP_URV_STEP_K_STRIDE = 128
    DUALWAVE_SWP_URV_DC_AXIS0 = SMEM_N_RPT * SMEM_V_LINE_STRIDE
    DUALWAVE_SWP_URV_DC_AXIS1 = 32
    DUALWAVE_SWP_URV_I5_STRIDE = D_128B_SIZE

    _lds_elem_dtype = _dtype_to_elem_type(dtype_str)

    LDS_KV_SIZE = LDS_KV_TOTAL_SIZE
    if BLOCK_LISTS:
        # build_block_union's int32 block states use the K/V buffers.
        LDS_KV_SIZE = max(LDS_KV_SIZE, (2 * KV_BLOCK_CAP + 1) * 4 // ELEM_BYTES)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, LDS_KV_SIZE, 16]
        if BLOCK_LISTS:
            # Entries, two prefetch slots past the last one, and the count.
            block_union: fx.Array[fx.Int32, KV_BLOCK_CAP + 4, 16]

    # log2 units
    DUALWAVE_SWP_RESCALE_THRESHOLD = 8.0

    @flyc.jit
    def _forward(Q, K, V, LSE, MaxScores, O, block_metadata, capture_args):
        # The schedule is built around ds_read_b64_tr_b16 and gfx950 LDS sizing.
        gpu_arch = get_rocm_arch()
        if const_expr(gpu_arch != FLEX_ARCH):
            raise RuntimeError(
                f"dual-wave flex forward targets {FLEX_ARCH} only, got {gpu_arch}"
            )
        elem_dtype = _dtype_to_elem_type(dtype_str)
        fm_fast = _FASTMATH
        v4i32_type = Vec.make_type(4, fx.Int32)
        v4f16_type = Vec.make_type(4, elem_dtype)
        v8f16_type = Vec.make_type(8, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        mfma_pack_type = v8f16_type

        _MFMA_MASK = 0x008
        _VALU_MASK = 0x002
        _EXP_MASK = 0x400

        kv_seq_len_i32 = runtime_extent(KV_SEQ_LEN, K, 2)
        seq_len_v = fx.Index(runtime_extent(SEQ_LEN, Q, 2))
        kv_seq_len_v = fx.Index(kv_seq_len_i32)
        (q_strides_v, k_strides_v, v_strides_v, o_strides_v) = (
            tuple(fx.Index(entry) for entry in runtime_strides(stride, tensor, seq)[:3])
            for stride, tensor, seq in (
                (q_stride, Q, runtime_extent(SEQ_LEN, Q, 2)),
                (k_stride, K, kv_seq_len_i32),
                (v_stride, V, kv_seq_len_i32),
                (o_stride, O, runtime_extent(SEQ_LEN, Q, 2)),
            )
        )
        stride_q_n_v, stride_k_n_v, stride_v_n_v, stride_o_n_v = (
            strides[2]
            for strides in (q_strides_v, k_strides_v, v_strides_v, o_strides_v)
        )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        lds_kv_base_ptr = _lds_ptr_from_index(lds_kv_base_idx)

        lds_scope_names = ("lds_k0", "lds_k1", "lds_v0", "lds_v1")

        def _lds_scope(kind, buf_id):
            return f"lds_{kind}{buf_id}"

        def _lds_alias_scopes(name):
            return _lds_alias_scope_array([name])

        def _lds_noalias_scopes(name):
            return _lds_alias_scope_array(
                [scope_name for scope_name in lds_scope_names if scope_name != name]
            )

        h_idx = fx.Index(getattr(gpu.block_idx, HEAD_AXIS))
        q_block_idx = fx.Index(getattr(gpu.block_idx, BLOCK_AXIS))
        batch_idx = fx.Index(gpu.block_idx.z)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane_mod_32 = lane % 32
        lane_div_32 = lane // 32

        _tid_i32 = _raw(fx.Int32(tid))
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32, arith.divsi(_tid_i32, _raw(fx.Int32(WARP_SIZE)))
        )
        if const_expr(STAGGER):
            _stagger_i32 = arith.divsi(_wave_id_uni_i32, _raw(fx.Int32(4)))
        wave_id_uni = fx.Index(_wave_id_uni_i32)

        wave_q_offset = wave_id * ROWS_PER_WAVE
        q_start = q_block_idx * BLOCK_M

        h_kv_idx = h_idx % NUM_HEADS_KV
        group_id = h_idx // NUM_HEADS_KV
        q_head_idx = h_kv_idx * GQA_GROUP_SIZE + group_id
        kv_head_idx = h_kv_idx

        kv_batch_idx = batch_idx
        if const_expr(kv_broadcast):
            kv_batch_idx = fx.Index(0)
        seqlen_q_v = seq_len_v

        def _row_base(strides, batch, head):
            return batch * strides[0] + head * strides[1]

        q_base = _row_base(q_strides_v, batch_idx, q_head_idx)
        o_base = _row_base(o_strides_v, batch_idx, q_head_idx)
        k_gmem_elem_offset = _row_base(k_strides_v, kv_batch_idx, kv_head_idx)
        v_gmem_elem_offset = _row_base(v_strides_v, kv_batch_idx, kv_head_idx)
        q_gmem_elem_offset = q_base + q_start * stride_q_n_v
        # Each (b, h) slab ends at row S, so rows past it load 0 and their
        # stores are dropped instead of touching another batch or head.
        q_end = q_base + seq_len_v * stride_q_n_v
        o_end = o_base + seq_len_v * stride_o_n_v
        k_end = k_gmem_elem_offset + kv_seq_len_v * stride_k_n_v
        v_end = v_gmem_elem_offset + kv_seq_len_v * stride_v_n_v

        if const_expr(HAS_MODS):
            mask_strides, mask_sizes = resolve_capture_extents(
                mask_buffer_strides, mask_buffer_sizes, capture_args, capture_table
            )
            mask_buffers, score_mod_buffers = make_capture_buffers(
                make_global_view,
                mask_buffer_count,
                mask_sizes,
                capture_table,
                capture_args,
            )
            score_mod_readers = make_score_mod_readers(
                make_global_view, score_mod_aux, score_mod_buffers
            )
            batch_i32 = fx.Int32(batch_idx)
            q_head_i32 = fx.Int32(q_head_idx)
            evaluate_mask = make_mask_evaluator(
                mask_program,
                mask_program_output,
                mask_strides,
                mask_buffers,
                lambda view, index: fx.Int32(view[index]),
                batch_i32,
                q_head_i32,
            )

        if const_expr(BLOCK_LISTS):
            union_ptr = _lds_ptr_from_index(fx.Index(fx.ptrtoint(lds.block_union.ptr)))

            def _union_entry(index):
                # The pipeline's K/V DMA writes must not order the entry reads.
                entry = llvm.LoadOp(
                    T.i32,
                    _lds_i32_ptr(union_ptr, index),
                    alignment=4,
                    alias_scopes=_lds_alias_scope_array(["lds_blocks"]),
                    noalias_scopes=_lds_alias_scope_array(list(lds_scope_names)),
                ).result
                return fx.Int32(rocdl.readfirstlane(T.i32, entry))

            def _entry_state(entry):
                if const_expr(not STAGGER):
                    return (fx.Int32(entry) >> fx.Int32(_STATE_SHIFT)) & fx.Int32(3)
                shift = fx.Int32(_STATE_SHIFT) + fx.Int32(_stagger_i32) * fx.Int32(2)
                return (fx.Int32(entry) >> shift) & fx.Int32(3)

            def _entry_kv_start(entry, half):
                block = fx.Int32(entry) & fx.Int32(_BLOCK_INDEX_MASK)
                return block * fx.Int32(2 * BLOCK_N) + fx.Int32(half * BLOCK_N)

            def _build_block_union():
                kv_counts, kv_lists, full_counts, full_lists = block_metadata
                q_blocks = fx.Int32(runtime_extent(num_q_blocks, kv_counts, 2))
                mask_rows = (
                    fx.Int32(runtime_extent(block_mask_batch, kv_counts, 0))
                    * fx.Int32(block_mask_heads)
                    * q_blocks
                )
                lists = []
                for counts, indices, width in (
                    (kv_counts, kv_lists, max_partial_blocks),
                    (full_counts, full_lists, max_full_blocks),
                ):
                    width = fx.Int32(runtime_extent(width, indices, 3))
                    lists.append(
                        (
                            make_global_view(counts, None, mask_rows, 1),
                            make_global_view(indices, None, mask_rows * width, 1),
                            width,
                        )
                    )
                if const_expr(block_mask_batch == 1):
                    mask_batch = fx.Int32(0)
                else:
                    mask_batch = batch_i32
                if const_expr(block_mask_heads == 1):
                    mask_head = fx.Int32(0)
                elif const_expr(block_mask_heads == NUM_HEADS_KV):
                    mask_head = fx.Int32(kv_head_idx)
                else:
                    mask_head = q_head_i32
                head_row = mask_batch * fx.Int32(block_mask_heads) + mask_head
                if const_expr(STAGGER):
                    q_block = fx.Int32(q_block_idx) * fx.Int32(2)
                    first_row = head_row * q_blocks + q_block
                    second_valid = q_block + fx.Int32(1) < q_blocks
                else:
                    first_row = head_row * q_blocks + fx.Int32(q_block_idx)
                    second_valid = None
                return build_block_union(
                    lists,
                    first_row,
                    second_valid,
                    fx.Int32(ceil_div(kv_seq_len_i32, 2 * BLOCK_N)),
                    lds_kv_base_ptr,
                    union_ptr,
                    KV_BLOCK_CAP,
                    BLOCK_SIZE,
                )

        DMA_BYTES = 16
        K_DMA_ISSUES = kv_dma_issues(NUM_WAVES, SMEM_K_D_RPT)
        V_DMA_ISSUES = kv_dma_issues(NUM_WAVES, SMEM_V_D_RPT)
        NUM_DMA_K = len(K_DMA_ISSUES)
        NUM_DMA_V = len(V_DMA_ISSUES)

        q_div, k_div, v_div, o_div = (
            flat_buffer_view(tensor, _raw(end * ELEM_BYTES))
            for tensor, end in ((Q, q_end), (K, k_end), (V, v_end), (O, o_end))
        )
        lse_div = flat_buffer_view(LSE)
        max_div = flat_buffer_view(MaxScores)
        _load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        _store_atom_32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        _store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        _dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _stats_store_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        _o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        _lds_ptr_ty = fx.PointerType.get(elem_dtype.ir_type, 2, DMA_BYTES)

        def _stats_store_f32(stats_div, f32_val, elem_index):
            pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(
                fx.Int32
            )
            fx.memref_store_vec(pack, _stats_store_reg)
            fx.copy(
                _store_atom_32,
                _stats_store_reg,
                fx.slice(stats_div, (None, fx.Int32(elem_index))),
            )

        def _buffer_load_128(elem_index):
            return fly.copy_atom_call_ssa(
                [v4i32_type],
                _load_atom_128,
                fx.slice(q_div, (None, fx.Int32(elem_index))),
            )

        def _tile_div(src, end, stride_n, tile_start):
            # Rows past Sk must load zeros: the tile offset moves from soffset
            # into the descriptor, whose num_records then ends at row Sk.
            return tile_buffer_view(src, end, tile_start * stride_n, ELEM_BYTES)

        def _buffer_load_lds_128(src_div, lds_byte_addr, src_elem, soffset_elems):
            # The atom scales soffset by the element size and drops alias-scope metadata.
            lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(src_div, (None, fx.Int32(src_elem)))
            fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))

        def _buffer_store_128(pack_i32_vec, elem_index):
            fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
            fx.copy(
                _store_atom_128,
                _o_store_reg_128,
                fx.slice(o_div, (None, fx.Int32(elem_index))),
            )

        lane_in_warp = tid % WARP_SIZE
        n_in_warp = lane_in_warp // VEC_KV
        d_bucket = lane_in_warp % VEC_KV

        c_neg_inf = fx.Float32(float("-inf"))
        # Finite floor for the causal row-max so exp2(-inf - floor) == 0 (no NaN).
        c_neg_floor = fx.Float32(-3.0e38)
        c_zero_f = fx.Float32(0.0)
        # Scores, row max and LSE are all log2-domain.
        c_sm_scale_log2e = fx.Float32(scale_log2e)
        c_eight_f = fx.Float32(DUALWAVE_SWP_RESCALE_THRESHOLD)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        v32_elem_type = Vec.make_type(PV_K_STEPS * 2 * 8, elem_dtype)
        v32f32_type = Vec.make_type(PV_K_STEPS * 2 * 8, fx.Float32)

        kv_tile_size = BLOCK_N
        # The pipeline needs an even tile count >= 4, so Sk pads to a multiple of 128.
        num_kv_tiles = fx.Index(2 * ceil_div(kv_seq_len_i32, 2 * BLOCK_N))
        if const_expr(CAUSAL):
            # Top-left causal: the q-block's last row is q_start + BLOCK_M - 1, so
            # tiles = (q_start + BLOCK_M) / 64 (even, >= 4), capped at the padded Sk.
            causal_num_tiles = (q_start + BLOCK_M) // kv_tile_size
            if const_expr(BLOCK_M < 4 * BLOCK_N):
                causal_num_tiles = fx.Index(
                    ArithValue(causal_num_tiles < fx.Index(4)).select(
                        fx.Index(4), causal_num_tiles
                    )
                )
            num_tiles = fx.Index(
                ArithValue(causal_num_tiles < num_kv_tiles).select(
                    causal_num_tiles, num_kv_tiles
                )
            )
        else:
            num_tiles = num_kv_tiles

        urk_base_per_lane = k_read_lane_offset(lane_mod_32, lane_div_32)

        urv_base_per_lane = (
            lane_div_32 * DUALWAVE_SWP_URV_GRPK
            + ((lane % 16) // 4) * DUALWAVE_SWP_URV_LANE_HI
            + ((lane // 16) % 2) * DUALWAVE_SWP_URV_GRP_N
            + (lane % 4) * DUALWAVE_SWP_URV_LANE_LO
        )

        _NEG_INF_F32_BITS = 0xFF800000

        _LGKMCNT_0_ONLY = 0xC07F

        def _cluster_barrier():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def _wait_cluster_barrier(vmcnt):
            rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
            _waitcnt_vm_n(vmcnt)
            _cluster_barrier()

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        # The softmax exponent must match the backward's unfused
        # exp2(S * scale_log2 - lse): no contract flag on either op.
        def _fmul_exact(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fx.arith.FastMathFlags.none)

        def _fsub_exact(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fx.arith.FastMathFlags.none)

        _mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, elem_dtype))

        def _mfma_acc(a, b, c):
            return fly.mma_atom_call_ssa([v16f32_type], _mma_atom, a, b, c)

        # MFMAs of one 64-key tile: QK^T, then PV.
        QK_MFMAS = 2 * K_STEPS_QK
        PV_MFMAS = 4 * D_CHUNKS

        def _sched_barrier_pairs(pairs, valu_cnt, group):
            for _ in range_constexpr(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_VALU_MASK, valu_cnt, group)

        def _sched_barrier_exp_pairs(pairs, exp_cnt, group):
            for _ in range_constexpr(pairs):
                rocdl.sched_group_barrier(_MFMA_MASK, 1, group)
                rocdl.sched_group_barrier(_EXP_MASK, exp_cnt, group)

        def _ds_read_tr_v4f16_imm(buf_id, lds_base_elem_idx, imm_bytes):
            # Without the scopes the backend drains the K/V DMA (vmcnt(0)) first.
            scope_name = _lds_scope("v", buf_id)
            ptr = _lds_byte_gep(lds_kv_base_ptr, lds_base_elem_idx * ELEM_BYTES)
            return _ds_read_tr16_b64_imm(
                v4f16_type,
                ptr,
                imm_bytes,
                _lds_alias_scopes(scope_name),
                _lds_noalias_scopes(scope_name),
            )

        def _global_idx_o(token_idx, col):
            return o_base + token_idx * stride_o_n_v + col

        def _concat_vectors(lhs, rhs):
            lhs_vec = Vec(lhs)
            rhs_vec = Vec(rhs)
            return lhs_vec.shuffle(
                rhs_vec,
                list(range(lhs_vec.numel))
                + [lhs_vec.numel + i for i in range(rhs_vec.numel)],
            )

        def _load_q_all(q_row_in_block):
            q_raw_packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = (ks * K_STEP_QK) + lane_div_32 * MFMA_LANE_K
                g_idx = q_row_in_block * stride_q_n_v + q_col
                q_i32_pack = _buffer_load_128(q_gmem_elem_offset + g_idx)
                q_raw_packs.append(
                    Vec(q_i32_pack, (4,), fx.Int32).bitcast(elem_dtype).ir_value()
                )
            q_all = _concat_pairwise(q_raw_packs, _concat_vectors)
            return Vec(q_all, (K_STEPS_QK * MFMA_LANE_K,), elem_dtype)

        def _get_q_pack(q_all, ks):
            q_vec = Vec(q_all)
            base = ks * MFMA_LANE_K
            return q_vec.shuffle(
                q_vec, [base + i for i in range(MFMA_LANE_K)]
            ).ir_value()

        def _bitcast_i32(value):
            return _raw(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _bitcast_f32(value):
            return _raw(ArithValue(value).bitcast(fx.Float32.ir_type))

        def _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
        ):
            asm_str = (
                f"v_cmp_lt_i32_e64 $0, $6, {int(thr_x)}\n\t"
                f"v_cmp_lt_i32_e64 $1, $6, {int(thr_y)}\n\t"
                "v_cndmask_b32_e64 $2, $4, $7, $0\n\t"
                "v_cndmask_b32_e64 $3, $5, $7, $1"
            )
            ret_struct_ty = ir.Type.parse("!llvm.struct<(i64, i64, i32, i32)>")
            ret = llvm.inline_asm(
                ret_struct_ty,
                [_raw(x_ref_i32), _raw(y_ref_i32), _raw(rel_i32), _raw(neg_inf_i32)],
                asm_str,
                "=s,=s,=v,=v,2,3,v,v,~{vcc}",
                has_side_effects=True,
            )
            return llvm.extractvalue(T.i32, ret, [2]), llvm.extractvalue(
                T.i32, ret, [3]
            )

        def _anchor_v_p(v_p):
            p_all = _v_p_to_vec32(v_p)
            return _v_vec32_to_p(
                llvm.inline_asm(p_all.type, [p_all], "", "=v,0", has_side_effects=True)
            )

        def _v_p_to_vec32(v_p):
            p_lo, p_hi = v_p
            p_lo_all = _concat_vectors(p_lo[0], p_lo[1])
            p_hi_all = _concat_vectors(p_hi[0], p_hi[1])
            return _concat_vectors(p_lo_all, p_hi_all).ir_value()

        def _v_vec32_to_p(v_p_all):
            p_vec = Vec(v_p_all, (PV_K_STEPS * 2 * 8,), elem_dtype)
            p_lo, p_hi = [], []
            for pks in range_constexpr(PV_K_STEPS):
                for packs, base in ((p_lo, pks * 8), (p_hi, (PV_K_STEPS + pks) * 8)):
                    packs.append(
                        p_vec.shuffle(p_vec, [base + i for i in range(8)]).ir_value()
                    )
            return p_lo, p_hi

        def _scale_v_p(v_p, scale_scalar):
            fm_fast_attr = ir.Attribute.parse("#llvm.fastmath<fast>")
            p_all = _v_p_to_vec32(v_p)
            if const_expr(dtype_str == "f16"):
                # Packed f16: the f32 round trip below spills the fp16 causal
                # main loop.
                scale_f16 = fx.Float32(scale_scalar).to(elem_dtype)
                scale_vec = Vec.from_elements([scale_f16], elem_dtype).broadcast_to(
                    PV_K_STEPS * 2 * 8
                )
                return _v_vec32_to_p(
                    arith.mulf(_raw(scale_vec), _raw(p_all), fastmath=fm_fast)
                )
            p_all_f32_op = llvm.FPExtOp(v32f32_type, _raw(p_all))
            p_all_f32_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(
                PV_K_STEPS * 2 * 8
            )
            p_scaled_f32 = arith.mulf(
                _raw(scale_vec), _raw(p_all_f32_op.result), fastmath=fm_fast
            )
            p_scaled_elem_op = llvm.FPTruncOp(v32_elem_type, p_scaled_f32)
            p_scaled_elem_op.operation.attributes["fastmathFlags"] = fm_fast_attr
            return _v_vec32_to_p(p_scaled_elem_op.result)

        @flyc.jit
        def _stagger_extra_barrier_if_one():
            """Emit `sched_barrier(0); s_barrier;` only when stagger == 1."""
            if fx.Int32(_stagger_i32) != fx.Int32(0):
                rocdl.sched_barrier(0)
                rocdl.s_barrier()

        def _stagger_extra_barrier_if_zero():
            """Emit `s_barrier;` only when stagger == 0."""
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [_stagger_i32],
                ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
                "s,~{scc}",
                has_side_effects=True,
            )

        def _trunc_pack_v8(f32_vals):
            if const_expr(dtype_str == "bf16"):
                pairs = []
                for j in range_constexpr(4):
                    pairs.append(
                        rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1])
                    )
                return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()
            f16_vals = []
            for i in range_constexpr(8):
                f16_vals.append(fx.Float32(f32_vals[i]).to(elem_dtype))
            return Vec.from_elements(f16_vals, elem_dtype).ir_value()

        def _k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return DUALWAVE_SWP_K_BUF_BASE[buf_id]
            return buf_id * DUALWAVE_SWP_KV_PER_BUFFER

        def _v_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return DUALWAVE_SWP_V_BUF_BASE[buf_id]
            return SMEM_K_TILE_ELEMS + buf_id * DUALWAVE_SWP_KV_PER_BUFFER

        def _async_load_kv(is_v, tile_start, buf_id, bounded):
            src, src_div = (V, v_div) if is_v else (K, k_div)
            stride = stride_v_n_v if is_v else stride_k_n_v
            gmem_offset = v_gmem_elem_offset if is_v else k_gmem_elem_offset
            line_stride = SMEM_V_LINE_STRIDE if is_v else SMEM_K_LINE_STRIDE
            lds_base = (_v_buf_base if is_v else _k_buf_base)(buf_id)
            lds_byte_base = lds_kv_base_idx + lds_base * ELEM_BYTES
            if const_expr(bounded):
                src_div = _tile_div(src, v_end if is_v else k_end, stride, tile_start)
            for line, key, column in V_DMA_ISSUES if is_v else K_DMA_ISSUES:
                lds_addr = (
                    lds_byte_base
                    + wave_id_uni * (line_stride * ELEM_BYTES)
                    + (line * line_stride * ELEM_BYTES)
                )
                n_in_tile = n_in_warp * SMEM_N_RPT + wave_id
                if const_expr(key):
                    n_in_tile = n_in_tile + key
                global_d = d_bucket * VEC_KV + column
                src_elem = gmem_offset + n_in_tile * stride + global_d
                soffset = 0 if bounded else tile_start * stride
                _buffer_load_lds_128(src_div, lds_addr, src_elem, soffset)

        def _async_load_k(tile_start, buf_id, bounded=False):
            _async_load_kv(False, tile_start, buf_id, bounded)

        def _async_load_v(tile_start, buf_id, bounded=False):
            _async_load_kv(True, tile_start, buf_id, bounded)

        def _reduction_pair(v_f32):
            v_i32 = _bitcast_i32(v_f32)
            pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
            swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
            lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
            rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
            return _bitcast_f32(lhs_i32), _bitcast_f32(rhs_i32)

        def _async_load_k_from_lds_to_vgpr(buf_id, urk_base):
            k_base = _k_buf_base(buf_id)
            k_lo = [None] * K_STEPS_QK
            k_hi = [None] * K_STEPS_QK

            def _load_k_pack_aligned(elem_idx):
                scope_name = _lds_scope("k", buf_id)
                byte_offset = elem_idx * ELEM_BYTES
                ptr = _lds_byte_gep(lds_kv_base_ptr, byte_offset)
                return llvm.LoadOp(
                    mfma_pack_type,
                    ptr,
                    alignment=16,
                    alias_scopes=_lds_alias_scopes(scope_name),
                    noalias_scopes=_lds_noalias_scopes(scope_name),
                ).result

            for ks in range_constexpr(K_STEPS_QK):
                idx_lo = k_base + urk_base + k_read_step_offset(ks)
                idx_hi = idx_lo + DUALWAVE_SWP_URK_N_STRIP_STRIDE
                k_lo[ks] = _load_k_pack_aligned(idx_lo)
                k_hi[ks] = _load_k_pack_aligned(idx_hi)
            return (k_lo, k_hi)

        def _read_v_packs_for_buf(buf_id, urv_base):
            v_base = _v_buf_base(buf_id)
            lds_base = v_base + urv_base
            packs = [[None] * D_CHUNKS for _ in range(4)]
            for dc in range_constexpr(D_CHUNKS):
                i_0 = dc // 2
                i_1 = dc % 2
                dc_off = (
                    i_0 * DUALWAVE_SWP_URV_DC_AXIS0 + i_1 * DUALWAVE_SWP_URV_DC_AXIS1
                )
                for k_substep in range_constexpr(4):
                    step_k_off = k_substep * DUALWAVE_SWP_URV_STEP_K_STRIDE
                    imm_lo = (step_k_off + dc_off) * ELEM_BYTES
                    a = _ds_read_tr_v4f16_imm(buf_id, lds_base, imm_lo)
                    b = _ds_read_tr_v4f16_imm(
                        buf_id,
                        lds_base,
                        imm_lo + DUALWAVE_SWP_URV_I5_STRIDE * ELEM_BYTES,
                    )
                    packs[k_substep][dc] = (
                        Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
                    )
            return packs

        def _mma0(v_k):
            k_lo, k_hi = v_k
            v_s_lo = c_zero_v16f32
            v_s_hi = c_zero_v16f32
            for ks in range_constexpr(K_STEPS_QK):
                q_pack = _get_q_pack(q_all_elem, ks)
                v_s_lo = _mfma_acc(k_lo[ks], q_pack, v_s_lo)
                v_s_hi = _mfma_acc(k_hi[ks], q_pack, v_s_hi)
            if const_expr(has_score_mod):
                # _score_mod_tile scales the scores itself.
                return (v_s_lo, v_s_hi)
            scale_vec = Vec.from_elements([c_sm_scale_log2e], fx.Float32).broadcast_to(
                16
            )
            return (_fmul_exact(v_s_lo, scale_vec), _fmul_exact(v_s_hi, scale_vec))

        def _read_lane_id():
            return fx.Int32(
                llvm.inline_asm(
                    T.i32,
                    [],
                    "v_mbcnt_lo_u32_b32 $0, -1, 0\n\tv_mbcnt_hi_u32_b32 $0, -1, $0",
                    "=v",
                    has_side_effects=True,
                )
            )

        def _tile_rows_and_keys(kv_start_i32):
            # The lane id is read anew for every tile: values the mods derive
            # from it would otherwise stay live across the loop and spill Q.
            tile_lane = _read_lane_id()
            row = fx.Int32(q_start + wave_id_uni * ROWS_PER_WAVE) + (
                tile_lane & fx.Int32(31)
            )
            key_base = fx.Int32(kv_start_i32) + (tile_lane >> fx.Int32(5)) * fx.Int32(4)
            return row, key_base

        def _tile_key(key_base, half, r):
            return key_base + fx.Int32(32 * half + _TILE_KEY_OFFSETS[r])

        def _score_mod_tile(v_s, row, key_base):
            # As in build_flex_attn_fwd_module: score_mod in natural units and
            # no contraction, so the backward recomputes the same exponent.
            modded = ([], [])
            with arith.fastmath(None):
                for half in range_constexpr(2):
                    for r in range_constexpr(16):
                        score = fx.Float32(v_s[half][r]) * fx.Float32(scale)
                        score = fx.Float32(
                            score_mod(
                                score,
                                batch_i32,
                                q_head_i32,
                                row,
                                _tile_key(key_base, half, r),
                                score_mod_readers,
                            )
                        )
                        modded[half].append(score * fx.Float32(_LOG2E))
            return modded

        @flyc.jit
        def _mask_block_tile(v_s, row, key_base, state):
            s_lo, s_hi = v_s
            if fx.Int32(state) < fx.Int32(_STATE_FULL):
                lists = _v_s_vec_to_lists(v_s)
                partial = fx.Int32(state) == fx.Int32(_STATE_PARTIAL)
                for half in range_constexpr(2):
                    for r in range_constexpr(16):
                        keep = partial
                        if const_expr(bool(mask_program)):
                            keep = keep & evaluate_mask(
                                row, _tile_key(key_base, half, r)
                            )
                        lists[half][r] = ArithValue(keep).select(
                            lists[half][r], c_neg_inf
                        )
                s_lo, s_hi = _lists_to_v_s(lists)
            return s_lo, s_hi

        def _apply_mods(v_s, kv_start_i32, entry=None):
            """score_mod and the BlockMask of one tile's scores (vectors)."""
            row, key_base = _tile_rows_and_keys(kv_start_i32)
            if const_expr(has_score_mod):
                v_s = _lists_to_v_s(
                    _score_mod_tile(_v_s_vec_to_lists(v_s), row, key_base)
                )
            if const_expr(BLOCK_LISTS):
                v_s = _mask_block_tile(v_s, row, key_base, _entry_state(entry))
            return v_s

        def _causal_mask_inplace(v_s, tile_idx, last_key_i32=None, kv_start_i32=None):
            s_lo, s_hi = v_s
            if const_expr(kv_start_i32 is None):
                kv_tile_start = tile_idx * BLOCK_N
                kv_start_i32 = fx.Int32(kv_tile_start)
            lane_off_i32 = fx.Int32(lane_div_32) * fx.Int32(4)
            # Keep key col <= last_key (q_row: top-left causal).
            if const_expr(last_key_i32 is None):
                last_key_i32 = q_row_i32
            rel_lo_i32 = fx.Int32(last_key_i32 - kv_start_i32 - lane_off_i32)
            rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
            neg_inf_i32 = fx.Int32(_NEG_INF_F32_BITS)

            for scores, rel_i32 in ((s_lo, rel_lo_i32), (s_hi, rel_hi_i32)):
                for x in range_constexpr(0, 16, 2):
                    new_x, new_y = _attn_mask_vec2_imm(
                        rel_i32,
                        neg_inf_i32,
                        _TILE_KEY_OFFSETS[x],
                        _TILE_KEY_OFFSETS[x + 1],
                        _bitcast_i32(scores[x]),
                        _bitcast_i32(scores[x + 1]),
                    )
                    scores[x] = _bitcast_f32(new_x)
                    scores[x + 1] = _bitcast_f32(new_y)

        def _v_s_vec_to_lists(v_s):
            return tuple([Vec(half)[r] for r in range_constexpr(16)] for half in v_s)

        def _lists_to_v_s(lists):
            return tuple(
                Vec.from_elements([_raw(v) for v in values], fx.Float32).ir_value()
                for values in lists
            )

        def _v_pair_to_vec32(v):
            return _concat_vectors(v[0], v[1]).ir_value()

        def _v_vec32_to_pair(v):
            v_vec = Vec(v, (32,), fx.Float32)
            v_lo = v_vec.shuffle(v_vec, list(range(16))).ir_value()
            v_hi = v_vec.shuffle(v_vec, [16 + i for i in range(16)]).ir_value()
            return v_lo, v_hi

        @flyc.jit
        def _causal_mask_prologue_if_needed(
            v_s, tile_idx=fx.Index(0), kv_end_pos=BLOCK_N
        ):
            s_lo, s_hi = v_s
            if q_start_pos_i32 < fx.Int32(kv_end_pos):
                lists = _v_s_vec_to_lists(v_s)
                _causal_mask_inplace(lists, tile_idx)
                s_lo, s_hi = _lists_to_v_s(lists)
            return s_lo, s_hi

        def _mask_tile(v_s, tile_idx, kv_end_pos, past_kv_end, entry=None, half=0):
            kv_start_i32 = None
            if const_expr(HAS_MODS):
                if const_expr(BLOCK_LISTS):
                    kv_start_i32 = _entry_kv_start(entry, half)
                    v_s = _apply_mods(v_s, kv_start_i32, entry)
                else:
                    v_s = _apply_mods(v_s, fx.Int32(tile_idx * BLOCK_N))
            if const_expr(past_kv_end):
                last_key_i32 = fx.Int32(kv_seq_len_i32 - 1)
                if const_expr(CAUSAL):
                    last_key_i32 = fx.Int32(
                        ArithValue(q_row_i32 < last_key_i32).select(
                            q_row_i32, last_key_i32
                        )
                    )
                v_s = _v_s_vec_to_lists(v_s)
                _causal_mask_inplace(v_s, tile_idx, last_key_i32, kv_start_i32)
                return v_s
            if const_expr(CAUSAL):
                return _causal_mask_prologue_if_needed(v_s, tile_idx, kv_end_pos)
            return _v_s_vec_to_lists(v_s)

        def _attn_row_max(v_s):
            m = c_neg_inf
            for half in v_s:
                for r in range_constexpr(16):
                    m = _fmax(m, half[r])
            lhs, rhs = _reduction_pair(m)
            return _fmax(lhs, rhs)

        def _mma1_step_k(step, v_p, v_v, v_o):
            v_p_lo, v_p_hi = v_p
            v_pk = v_v[step]
            if const_expr(step < 2):
                p_pk = v_p_lo[step]
            else:
                p_pk = v_p_hi[step - 2]
            for dc in range_constexpr(D_CHUNKS):
                v_o[dc] = _mfma_acc(v_pk[dc], p_pk, v_o[dc])
            return v_o

        def _mma1(v_p, v_v, v_o):
            for step in range_constexpr(4):
                v_o = _mma1_step_k(step, v_p, v_v, v_o)
            return v_o

        def _attn_sub_row(v_s, row_max):
            subs = [
                [_fsub_exact(half[r], row_max) for r in range_constexpr(16)]
                for half in v_s
            ]
            return tuple(Vec.from_elements(sub, fx.Float32).ir_value() for sub in subs)

        def _exp2_wait_state(values):
            # v_exp_f32 needs one wait state before a VALU reads its result, and
            # LLVM does not insert it on gfx950. The tied operands order every
            # consumer after the nop without splitting the scheduling region.
            ret = llvm.inline_asm(
                ir.Type.parse(f"!llvm.struct<({', '.join(['f32'] * len(values))})>"),
                values,
                "s_nop 0",
                ",".join(["=v"] * len(values) + [str(i) for i in range(len(values))]),
                has_side_effects=False,
            )
            return [llvm.extractvalue(T.f32, ret, [i]) for i in range(len(values))]

        def _attn_exp2_slice(v_s, start):
            if const_expr(start == 0):
                s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
                lo_partial = _exp2_wait_state(
                    [rocdl.exp2(T.f32, _raw(s_lo[r])) for r in range_constexpr(16)]
                )
                return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]

            lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            hi_full = _exp2_wait_state(
                [rocdl.exp2(T.f32, _raw(Vec(v_s[1])[r])) for r in range_constexpr(16)]
            )
            return lo_partial, hi_full

        def _attn_sum(v_p):
            local_sum = c_zero_f
            for half in v_p:
                for r in range_constexpr(16):
                    local_sum = _fadd(local_sum, half[r])
            lhs_sum, rhs_sum = _reduction_pair(local_sum)
            return _fadd(lhs_sum, rhs_sum)

        def _tie_pack(pack):
            return tied_nop(Vec(pack), 1, has_side_effects=False).ir_value()

        def _cast_p(v_p):
            # Two wait states between the converts and the PV MFMA reading them.
            packs = ([], [])
            for pks in range_constexpr(PV_K_STEPS):
                for half, values in zip(packs, v_p):
                    half.append(
                        _tie_pack(_trunc_pack_v8(values[pks * 8 : pks * 8 + 8]))
                    )
            return packs

        def _sum_and_pack_p(v_p, l_row):
            v_p = _attn_exp2_slice(v_p, 16)
            l_row = _fadd(l_row, _attn_sum(v_p))
            return _anchor_v_p(_cast_p(v_p)), l_row

        def _epilogue_softmax(v_s, m_row):
            row_max = _fmax(m_row, _attn_row_max(v_s))
            rescale = rocdl.exp2(T.f32, _raw(_fsub(m_row, row_max)))
            return _attn_exp2_slice(_attn_sub_row(v_s, row_max), 0), row_max, rescale

        def _scale_o(v_o, scale_scalar):
            scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
            for dc in range_constexpr(D_CHUNKS):
                v_o[dc] = _fmul(Vec(v_o[dc]), scale_vec)

        def _anchor_v_o(v_o):
            """Pin v_o accumulators at the current source position."""
            acc_irs = [_raw(v_o[dc]) for dc in range_constexpr(D_CHUNKS)]
            ret_ty = ir.Type.parse(
                f"!llvm.struct<({', '.join(['vector<16xf32>'] * D_CHUNKS)})>"
            )
            ret = llvm.inline_asm(
                ret_ty,
                acc_irs,
                "",
                ",".join(["=v"] * D_CHUNKS + [str(dc) for dc in range(D_CHUNKS)]),
                has_side_effects=True,
            )
            return [
                llvm.extractvalue(acc_irs[dc].type, ret, [dc])
                for dc in range_constexpr(D_CHUNKS)
            ]

        def _anchor_scalar_f32(x):
            """Pin a scalar f32 at the current source position (no-op asm)."""
            x_ir = _raw(x)
            return llvm.inline_asm(x_ir.type, [x_ir], "", "=v,0", has_side_effects=True)

        @flyc.jit
        def _lazy_rescale_o(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = _fsub(m_tile_max, m_row)
            below = ArithValue(fx.Float32(m_diff) <= c_eight_f)
            ballot = rocdl.ballot(T.i64, _raw(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq, _raw(ballot), _read_exec_i64()
            )
            all_below = llvm.intr_expect(
                all_below, arith.constant(1, type=ir.IntegerType.get_signless(1))
            )

            o_out = [_raw(v_o[dc]) for dc in range_constexpr(D_CHUNKS)]
            m_out = _raw(m_row)
            l_out = _raw(l_row)
            vp_out = _v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                # Lanes whose tile max is below their running max keep the running
                # max; otherwise exp2(m_row - m_tile_max) can overflow.
                m_new = _fmax(m_row, m_tile_max)
                corr = rocdl.exp2(T.f32, _raw(_fsub(m_row, m_new)))
                scaled_accs = list(v_o)
                _scale_o(scaled_accs, corr)
                o_out = [_raw(scaled_accs[dc]) for dc in range_constexpr(D_CHUNKS)]
                scaled_p = _scale_v_p(v_p, corr)
                if const_expr(not STAGGER and dtype_str == "bf16"):
                    # The PV MFMAs after the branch back may read the converts'
                    # registers one wait state later; step 0 read its pack before.
                    p_lo, p_hi = scaled_p
                    scaled_p = (
                        [p_lo[0], *(_tie_pack(p) for p in p_lo[1:])],
                        [_tie_pack(p) for p in p_hi],
                    )
                vp_out = _v_p_to_vec32(scaled_p)
                l_out = _raw(_fmul(l_row, corr))
                m_out = _anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out, _v_vec32_to_p(vp_out))

        if const_expr(BLOCK_LISTS):
            num_tiles = fx.Index(_build_block_union() * fx.Int32(2))
            entry_0 = _union_entry(0)
            entry_1 = _union_entry(1)
            first_tiles = (
                _entry_kv_start(entry_0, 0),
                _entry_kv_start(entry_0, 1),
                _entry_kv_start(entry_1, 0),
            )
            first_tiles = tuple(fx.Index(start) for start in first_tiles)
        else:
            first_tiles = (0, BLOCK_N, 2 * BLOCK_N)
        _async_load_k(first_tiles[0], 0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        q_row_in_block = wave_q_offset + lane_mod_32
        q_start_pos_i32 = fx.Int32(q_start + wave_id_uni * ROWS_PER_WAVE)
        q_row = q_start + q_row_in_block
        q_row_i32 = fx.Int32(q_row)
        q_all_elem = _load_q_all(q_row_in_block)

        _async_load_k(first_tiles[1], 1)
        _async_load_v(first_tiles[0], 0)
        v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _waitcnt_vm_n(NUM_DMA_V)

        if const_expr(STAGGER):
            # Group B (waves 4-7) takes one extra s_barrier, putting the groups one cluster apart.
            _stagger_extra_barrier_if_one()

        v_s_0 = _mma0(v_k)
        rocdl.sched_barrier(0)
        if const_expr(BLOCK_LISTS):
            v_s_0 = _apply_mods(v_s_0, _entry_kv_start(entry_0, 0), entry_0)
        elif const_expr(HAS_MODS):
            v_s_0 = _apply_mods(v_s_0, fx.Int32(0))
        if const_expr(CAUSAL):
            v_s_0 = _causal_mask_prologue_if_needed(v_s_0)
        else:
            v_s_0 = _v_s_vec_to_lists(v_s_0)
        m_row_pro = _attn_row_max(v_s_0)
        if const_expr(CAUSAL or HAS_MODS):
            # Flooring the init value keeps every later max finite.
            m_row_pro = _fmax(m_row_pro, c_neg_floor)
        v_s_0 = _attn_sub_row(v_s_0, m_row_pro)
        v_p_0 = _attn_exp2_slice(v_s_0, 0)
        _cluster_barrier()

        _async_load_k(first_tiles[2], 0, BLOCK_LISTS and MASK_SECOND_LAST_TILE)

        l_row_init = c_zero_f
        init_args = [m_row_pro, l_row_init]
        for _ in range_constexpr(D_CHUNKS):
            init_args.append(c_zero_v16f32)
        init_args.append(_v_pair_to_vec32(v_p_0))
        if const_expr(WRITE_MAX_SCORES):
            # m_row lags the true row max under lazy rescale; track it separately.
            init_args.append(m_row_pro)
        # The entries of the blocks of tiles j - 2 and j - 1 at iteration j.
        entry_args = 3 + D_CHUNKS + int(WRITE_MAX_SCORES)
        if const_expr(BLOCK_LISTS):
            init_args += [entry_0, entry_1]

        # Each iteration is 8 clusters alternating memory (DMA issue, LDS reads,
        # s_waitcnt, s_barrier) and compute (MFMA, softmax) for two KV tiles.
        def _pipeline_iteration(j_idx, loop_args, bounded_k=False):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            v_p_0 = _v_vec32_to_pair(loop_args[2 + D_CHUNKS])
            if const_expr(WRITE_MAX_SCORES):
                m_exact = loop_args[3 + D_CHUNKS]
            if const_expr(BLOCK_LISTS):
                entry_prev = loop_args[entry_args]
                entry_cur = loop_args[entry_args + 1]
                tiles = (
                    _entry_kv_start(entry_prev, 1),
                    _entry_kv_start(entry_cur, 0),
                    _entry_kv_start(entry_cur, 1),
                )
                load_tiles = tuple(fx.Index(start) for start in tiles)
            elif const_expr(HAS_MODS):
                tiles = (
                    fx.Int32((j_idx - 2) * BLOCK_N),
                    fx.Int32((j_idx - 1) * BLOCK_N),
                )

            _async_load_v(load_tiles[0] if BLOCK_LISTS else (j_idx - 2) * BLOCK_N, 1)
            if const_expr(BLOCK_LISTS):
                entry_next = _union_entry(fx.Int32((j_idx + 1) // 2))
            v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
            _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

            v_s_1 = _mma0(v_k)
            v_p_0, l_row = _sum_and_pack_p(v_p_0, l_row)
            _sched_barrier_exp_pairs(6, 3, 1)
            _sched_barrier_pairs(QK_MFMAS - 6, 5, 1)
            _cluster_barrier()

            _async_load_k(load_tiles[2] if BLOCK_LISTS else j_idx * BLOCK_N, 1)
            if const_expr(HAS_MODS):
                # Before the V reads, whose registers would be live meanwhile.
                v_s_1 = _apply_mods(
                    v_s_1, tiles[0], entry_prev if BLOCK_LISTS else None
                )
                rocdl.sched_barrier(0)
            v_v = _read_v_packs_for_buf(0, urv_base_per_lane)
            _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

            rocdl.s_setprio(1)
            v_o = _mma1_step_k(0, v_p_0, v_v, v_o)
            # v_s_1 holds tile j_idx-2, strictly below the causal diagonal.
            v_s_1 = _v_s_vec_to_lists(v_s_1)
            m_tile_max_a = _attn_row_max(v_s_1)
            if const_expr(WRITE_MAX_SCORES):
                m_exact = _fmax(m_exact, m_tile_max_a)

            _sched_barrier_pairs(D_CHUNKS, 6, 2)

            v_o, m_row, l_row, v_p_0 = _lazy_rescale_o(
                v_o, m_row, l_row, m_tile_max_a, v_p_0
            )
            for step in range_constexpr(1, 4):
                v_o = _mma1_step_k(step, v_p_0, v_v, v_o)
            v_s_1 = _attn_sub_row(v_s_1, m_row)
            v_p_1 = _attn_exp2_slice(v_s_1, 0)

            _sched_barrier_pairs(3 * D_CHUNKS - 6, 6, 2)
            # Interleave each MFMA with 3 TRANS (exp2) ops.
            _sched_barrier_exp_pairs(6, 3, 2)
            rocdl.s_setprio(0)
            # sched_barrier(0) pins s_setprio and s_barrier at the cluster boundary.
            _cluster_barrier()

            _async_load_v(load_tiles[1] if BLOCK_LISTS else (j_idx - 1) * BLOCK_N, 0)
            v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
            _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

            v_s_0 = _mma0(v_k)
            v_p_1, l_row = _sum_and_pack_p(v_p_1, l_row)
            _sched_barrier_exp_pairs(6, 3, 3)
            _sched_barrier_pairs(QK_MFMAS - 6, 5, 3)
            _cluster_barrier()

            if const_expr(BLOCK_LISTS):
                next_tile = fx.Index(_entry_kv_start(entry_next, 0))
            else:
                next_tile = (j_idx + 1) * BLOCK_N
            _async_load_k(next_tile, 0, bounded_k)
            if const_expr(HAS_MODS):
                v_s_0 = _apply_mods(v_s_0, tiles[1], entry_cur if BLOCK_LISTS else None)
                if const_expr(CAUSAL):
                    v_s_0 = _causal_mask_prologue_if_needed(
                        v_s_0, j_idx - 1, j_idx * BLOCK_N
                    )
                rocdl.sched_barrier(0)
            v_packs_b = _read_v_packs_for_buf(1, urv_base_per_lane)
            if const_expr(HAS_MODS):
                v_s_0 = _v_s_vec_to_lists(v_s_0)
            elif const_expr(CAUSAL):
                v_s_0 = _causal_mask_prologue_if_needed(
                    v_s_0, j_idx - 1, j_idx * BLOCK_N
                )
            else:
                v_s_0 = _v_s_vec_to_lists(v_s_0)
            _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

            rocdl.s_setprio(1)
            v_o = _mma1_step_k(0, v_p_1, v_packs_b, v_o)
            m_tile_max_b = _attn_row_max(v_s_0)
            if const_expr(WRITE_MAX_SCORES):
                m_exact = _fmax(m_exact, m_tile_max_b)
            _sched_barrier_pairs(D_CHUNKS, 6, 4)

            v_o, m_row, l_row, v_p_1 = _lazy_rescale_o(
                v_o, m_row, l_row, m_tile_max_b, v_p_1
            )
            for step in range_constexpr(1, 4):
                v_o = _mma1_step_k(step, v_p_1, v_packs_b, v_o)
            v_s_0 = _attn_sub_row(v_s_0, m_row)
            v_p_0 = _attn_exp2_slice(v_s_0, 0)
            _sched_barrier_pairs(3 * D_CHUNKS - 6, 5, 4)
            _sched_barrier_exp_pairs(6, 3, 4)
            rocdl.s_setprio(0)
            _cluster_barrier()

            yield_args = [m_row, l_row] + v_o + [_v_pair_to_vec32(v_p_0)]
            if const_expr(WRITE_MAX_SCORES):
                yield_args.append(m_exact)
            if const_expr(BLOCK_LISTS):
                yield_args += [entry_cur, entry_next]
            return yield_args

        # The last iteration loads K tile num_tiles - 2; peel it when that tile
        # crosses Sk so the steady-state loop keeps soffset-based loads.
        if const_expr(not PEEL_LAST_ITERATION):
            loop_end = num_tiles - fx.Index(1)
        elif const_expr(CAUSAL):
            reaches_kv_end = ArithValue(num_tiles == num_kv_tiles)
            loop_end = fx.Index(
                reaches_kv_end.select(num_tiles - fx.Index(3), num_tiles - fx.Index(1))
            )
        else:
            loop_end = num_tiles - fx.Index(3)
        loop_results = init_args
        for j, loop_args in range(fx.Index(3), loop_end, fx.Index(2), init=init_args):
            loop_results = yield _pipeline_iteration(j, loop_args, BOUND_LOOP_K)
        if const_expr(PEEL_LAST_ITERATION):
            if const_expr(CAUSAL):
                if reaches_kv_end:
                    loop_results = _pipeline_iteration(
                        num_tiles - fx.Index(3), loop_results, True
                    )
            else:
                loop_results = _pipeline_iteration(
                    num_tiles - fx.Index(3), loop_results, True
                )

        # Epilogue: the last three tiles, with unconditional rescales.
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]
        v_p_0 = _v_vec32_to_pair(loop_results[2 + D_CHUNKS])
        if const_expr(WRITE_MAX_SCORES):
            m_exact = loop_results[3 + D_CHUNKS]

        max_m3 = num_tiles - 3
        max_m2 = num_tiles - 2
        max_m1 = num_tiles - 1
        entry_m3 = entry_m1 = None
        if const_expr(BLOCK_LISTS):
            entry_m3, entry_m1 = loop_results[entry_args : entry_args + 2]

        def _last_tile(tile_idx, entry, half):
            if const_expr(BLOCK_LISTS):
                return fx.Index(_entry_kv_start(entry, half))
            return tile_idx * BLOCK_N

        _async_load_v(_last_tile(max_m3, entry_m3, 1), 1)
        v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

        v_s_1 = _mma0(v_k)
        v_p_0, l_row = _sum_and_pack_p(v_p_0, l_row)
        _sched_barrier_exp_pairs(6, 3, 5)
        _sched_barrier_pairs(QK_MFMAS - 6, 5, 5)
        _cluster_barrier()

        _async_load_k(_last_tile(max_m1, entry_m1, 1), 1, MASK_LAST_TILE)
        v_packs_e3 = _read_v_packs_for_buf(0, urv_base_per_lane)
        v_s_1 = _mask_tile(v_s_1, max_m3, max_m2 * BLOCK_N, False, entry_m3, 1)
        _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

        rocdl.s_setprio(1)
        v_o = _mma1(v_p_0, v_packs_e3, v_o)
        v_p_1, m_row, rescale_e3 = _epilogue_softmax(v_s_1, m_row)
        _sched_barrier_pairs(PV_MFMAS - 6, 5, 6)
        _sched_barrier_exp_pairs(6, 3, 6)
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e3)
        v_o = _anchor_v_o(v_o)

        rocdl.s_setprio(0)
        _cluster_barrier()

        _async_load_v(_last_tile(max_m2, entry_m1, 0), 0, MASK_SECOND_LAST_TILE)
        v_k = _async_load_k_from_lds_to_vgpr(0, urk_base_per_lane)
        _wait_cluster_barrier(NUM_DMA_K + NUM_DMA_V)

        v_s_0 = _mma0(v_k)
        l_row = _fmul(l_row, rescale_e3)
        v_p_1, l_row = _sum_and_pack_p(v_p_1, l_row)
        _sched_barrier_exp_pairs(6, 3, 7)
        _sched_barrier_pairs(QK_MFMAS - 6, 5, 7)
        _cluster_barrier()

        v_packs_e7 = _read_v_packs_for_buf(1, urv_base_per_lane)
        v_s_0 = _mask_tile(
            v_s_0, max_m2, max_m1 * BLOCK_N, MASK_SECOND_LAST_TILE, entry_m1, 0
        )
        _wait_cluster_barrier(NUM_DMA_V)

        rocdl.s_setprio(1)
        v_o = _mma1(v_p_1, v_packs_e7, v_o)
        v_p_0, m_row, rescale_e7 = _epilogue_softmax(v_s_0, m_row)
        _sched_barrier_pairs(PV_MFMAS - 6, 5, 8)
        _sched_barrier_exp_pairs(6, 3, 8)
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e7)
        v_o = _anchor_v_o(v_o)
        rocdl.s_setprio(0)
        _cluster_barrier()

        _async_load_v(_last_tile(max_m1, entry_m1, 1), 1, MASK_LAST_TILE)
        v_k = _async_load_k_from_lds_to_vgpr(1, urk_base_per_lane)
        _wait_cluster_barrier(NUM_DMA_V)

        v_s_1 = _mma0(v_k)
        l_row = _fmul(l_row, rescale_e7)
        v_p_0, l_row = _sum_and_pack_p(v_p_0, l_row)
        _sched_barrier_exp_pairs(6, 3, 9)
        _sched_barrier_pairs(QK_MFMAS - 6, 5, 9)
        _cluster_barrier()

        v_packs_e11 = _read_v_packs_for_buf(0, urv_base_per_lane)
        v_s_1 = _mask_tile(
            v_s_1, max_m1, num_tiles * BLOCK_N, MASK_LAST_TILE, entry_m1, 1
        )
        _wait_cluster_barrier(0)

        v_o = _mma1(v_p_0, v_packs_e11, v_o)
        v_p_1, m_row, rescale_e11 = _epilogue_softmax(v_s_1, m_row)
        _sched_barrier_pairs(PV_MFMAS - 7, 6, 10)
        _sched_barrier_exp_pairs(7, 3, 10)
        rocdl.sched_barrier(0)
        v_p_1 = _attn_exp2_slice(v_p_1, 16)
        l_row = _fmul(l_row, rescale_e11)
        l_row = _fadd(l_row, _attn_sum(v_p_1))
        v_p_1 = _anchor_v_p(_cast_p(v_p_1))
        rocdl.sched_barrier(0)
        _scale_o(v_o, rescale_e11)
        v_o = _anchor_v_o(v_o)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        v_packs_e13 = _read_v_packs_for_buf(1, urv_base_per_lane)
        rocdl.s_waitcnt(_LGKMCNT_0_ONLY)
        _cluster_barrier()

        v_o = _mma1(v_p_1, v_packs_e13, v_o)

        inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
        has_values = ArithValue(fx.Float32(l_row) > c_zero_f)
        inv_l = has_values.select(inv_l_rcp, c_zero_f)

        # log2-domain LSE (output_stats_in_log2).
        lse_val = _fadd(
            fx.Float32(m_row),
            fx.Float32(fmath.log2(fx.Float32(l_row), fastmath=fm_fast)),
        )
        lse_val = has_values.select(lse_val, c_neg_inf)
        if const_expr(WRITE_MAX_SCORES):
            max_val = has_values.select(_fmax(m_exact, m_row), c_neg_inf)
        out_q_row, out_lane_div_32 = q_row, lane_div_32
        if const_expr(not STAGGER):
            # Store addresses sharing the Q loads' lane values keep those live
            # through the loops, which spills at (256, 256).
            out_lane = fx.Index(_read_lane_id())
            out_lane_div_32 = out_lane // 32
            out_q_row = q_start + wave_id_uni * ROWS_PER_WAVE + out_lane % 32
        # Lanes 0-31 and 32-63 hold the same rows after the half-wave reduction.
        stats_idx = (
            batch_idx * fx.Index(NUM_HEADS_Q) + q_head_idx
        ) * seqlen_q_v + out_q_row
        if ArithValue(out_lane_div_32 == fx.Index(0)):
            if ArithValue(out_q_row < seqlen_q_v):
                _stats_store_f32(lse_div, lse_val, stats_idx)
                if const_expr(WRITE_MAX_SCORES):
                    _stats_store_f32(max_div, max_val, stats_idx)

        _scale_o(v_o, inv_l)

        if const_expr(STAGGER):
            # Group A takes the matching extra s_barrier, realigning the groups.
            _stagger_extra_barrier_if_zero()

        # Each lane fuses its 4 columns with its half-wave partner's for dwordx4 stores.
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def _o_pack_2dw(dc, store_group):
            r_base = store_group * 4
            if const_expr(dtype_str == "bf16"):
                lo = rocdl.cvt_pk_bf16_f32(
                    Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1]
                )
                hi = rocdl.cvt_pk_bf16_f32(
                    Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3]
                )
                return lo, hi
            o_f16 = []
            for i in range_constexpr(4):
                o_f16.append(fx.Float32(Vec(v_o[dc])[r_base + i]).to(elem_dtype))
            pack = Vec.from_elements(o_f16, elem_dtype).bitcast(fx.Int32)
            return _raw(pack[0]), _raw(pack[1])

        is_hi_half = ArithValue(out_lane_div_32 != fx.Index(0))

        def _swap_halves(dw):
            # permlane32_swap(a,b) -> (a.lo|b.lo, a.hi|b.hi); with a=b=dw the
            # partner dword dw[lane^32] is result[1] on low lanes, [0] on high.
            swapped = rocdl.permlane32_swap(
                pair_i32_ty, _raw(dw), _raw(dw), False, False
            )
            lo_res = llvm.extractvalue(T.i32, swapped, [0])
            hi_res = llvm.extractvalue(T.i32, swapped, [1])
            return is_hi_half.select(lo_res, hi_res)

        for dc in range_constexpr(D_CHUNKS):
            for g in range_constexpr(2):
                d0_a, d1_a = _o_pack_2dw(dc, 2 * g)
                d0_b, d1_b = _o_pack_2dw(dc, 2 * g + 1)
                y0_a, y1_a = _swap_halves(d0_a), _swap_halves(d1_a)
                y0_b, y1_b = _swap_halves(d0_b), _swap_halves(d1_b)
                w0 = is_hi_half.select(y0_b, _raw(d0_a))
                w1 = is_hi_half.select(y1_b, _raw(d1_a))
                w2 = is_hi_half.select(_raw(d0_b), y0_a)
                w3 = is_hi_half.select(_raw(d1_b), y1_a)
                o_pack = Vec.from_elements(
                    [fx.Int32(w0), fx.Int32(w1), fx.Int32(w2), fx.Int32(w3)], fx.Int32
                )
                d_col = (dc * D_CHUNK) + (2 * g + out_lane_div_32) * 8
                o_global = _global_idx_o(out_q_row, d_col)
                _buffer_store_128(o_pack, o_global)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flex_attn_fwd_dualwave_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        MaxScores: fx.Tensor,
        O: fx.Tensor,
    ):
        _forward(Q, K, V, LSE, MaxScores, O, None, None)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flex_attn_fwd_dualwave_mods_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        MaxScores: fx.Tensor,
        KvNumBlocks: fx.Tensor,
        KvIndices: fx.Tensor,
        FullKvNumBlocks: fx.Tensor,
        FullKvIndices: fx.Tensor,
        Capture0: fx.Tensor,
        Capture1: fx.Tensor,
        Capture2: fx.Tensor,
        Capture3: fx.Tensor,
        O: fx.Tensor,
    ):
        _forward(
            Q,
            K,
            V,
            LSE,
            MaxScores,
            O,
            (KvNumBlocks, KvIndices, FullKvNumBlocks, FullKvIndices),
            (Capture0, Capture1, Capture2, Capture3),
        )

    def kernel_attrs():
        # Parsed inside the launch's MLIR context.
        return {
            "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
            "passthrough": [["unsafe-fp-math", "true"]],
            # f32 denormals are zero. The backend reads this attribute, not the
            # denormal-fp-math-f32 passthrough string.
            "llvm.denormal_fpenv": ir.Attribute.parse(
                "#llvm.denormal_fpenv<default_output_mode = ieee, "
                "default_input_mode = ieee, float_output_mode = preservesign, "
                "float_input_mode = preservesign>"
            ),
        }

    # Ops without an explicit fastmath= inherit the ambient fast_fp_math scope.
    compile_hints = {
        "waves_per_eu": int(waves_per_eu),
        "fastmath": _FASTMATH,
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }
    if HAS_MODS:
        launch = _make_forward_launch(
            lambda *args: flex_attn_fwd_dualwave_mods_kernel(
                *args, value_attrs=kernel_attrs()
            ),
            grid=lambda query: tile_grid(
                NUM_HEADS_Q,
                ceil_div(runtime_extent(SEQ_LEN, query, 2), BLOCK_M),
                runtime_extent(batch_size, query, 0),
                HEAD_SLOW_ORDER,
            ),
            num_threads=BLOCK_SIZE,
            mask_buffer_count=(
                capture_slots
                if capture_table
                else mask_buffer_count + score_mod_buffer_count
            ),
            waves_per_eu=int(waves_per_eu),
            mod_key=mod_key,
        )
        launch.compile_hints = compile_hints
        return launch

    @flyc.jit
    def launch(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        logsumexp: fx.Tensor,
        max_scores: fx.Tensor,
        output: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        flex_attn_fwd_dualwave_kernel(
            query, key, value, logsumexp, max_scores, output, value_attrs=kernel_attrs()
        ).launch(
            grid=tile_grid(
                NUM_HEADS_Q,
                ceil_div(runtime_extent(SEQ_LEN, query, 2), BLOCK_M),
                runtime_extent(batch_size, query, 0),
                HEAD_SLOW_ORDER,
            ),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch.compile_hints = compile_hints
    return launch
