# mypy: allow-untyped-defs
"""Build choices of the FlyDSL flex kernels that depend on B, Sq and Sk.

Sizes may be SymInts: each choice is then one guard on a predicate built with
``&`` and ``|``, so a graph recompiles only when the choice changes.

The planner's guard set is intentionally limited to launch-choice boundaries:
the power-of-two bounds used for runtime block counts, the decode/prefill
boundary at ``Sq == 128``, the large-query head-order boundary at ``Sq ==
16384``, and the occupancy/workspace thresholds in the choice predicates.
These guards are created by evaluating SymBool predicates here rather than by
calling the generic sizevars guard helpers because the planner returns a
single, coherent choice dictionary.
"""

import functools
import operator

from . import flex_attn_target
from .flex_attn_bwd_utils import (
    backward_key_rows,
    choose_dq_partitions,
    dq_workspace_head_group,
    MASK_TRAVERSAL_BLOCK_LIST,
)


# Four-wave CTAs cut the launch count once the 128-row prefill grid has this
# many CTAs per CU.
_FOUR_WAVE_PREFILL_CTAS_PER_CU = 2
# Split-KV decode targets one CTA per CU, each of at least eight 128-key blocks.
_SPLIT_KV_MIN_BLOCKS = 8
_MAX_KV_SPLITS = 256
# gfx950 hands consecutive workgroups to its 8 XCDs in turn; with Hq a multiple
# of 8 and few query blocks per head, head-slow order puts every head on every
# XCD at once.
_XCDS = 8
_HEAD_SLOW_MIN_SEQ_Q = 16384
# The kernels stage 2-byte bf16/fp16 tiles.
_ELEMENT_BYTES = 2


def use_head_slow_order(num_q_heads: int, seq_q) -> bool:
    """Whether a forward without a BlockMask runs its grid head-slow."""
    return num_q_heads % _XCDS == 0 and bool(seq_q >= _HEAD_SLOW_MIN_SEQ_Q)


def _power_of_two_bound(*extents):
    """The least power of two >= 32 bounding ``extents``, one guard per step."""
    bound = 32
    while not bool(
        functools.reduce(operator.and_, (extent <= bound for extent in extents))
    ):
        bound *= 2
    return bound


def _is_wide_head(qk_head_dim, v_head_dim):
    # Past D192/V128, or with V above 128, the output and Q fragments overflow
    # the VGPRs of the 1-, 2- and 8-wave decode CTAs and of split-KV, so decode
    # runs as a 4-wave pipeline.
    return qk_head_dim + v_head_dim > 320 or v_head_dim > 128


def _select_owner_waves(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    num_cus: int,
) -> int:
    """Choose how many independent 32-row query owners share one CTA."""
    if seq_q < 128:
        if _is_wide_head(qk_head_dim, v_head_dim):
            return 4
        packed_query_rows = (num_q_heads // num_kv_heads) * seq_q
        # One wave per 32 packed rows, rounded up to a power of two.
        waves = next(
            (waves for waves in (1, 2, 4) if packed_query_rows <= 32 * waves), 8
        )
        # Every thread stages whole 16-byte packs of each 64-row K/V tile.
        while (qk_head_dim // 8) % waves or (v_head_dim // 8) % waves:
            waves //= 2
        return waves

    base_ctas = batch_size * num_q_heads * ((seq_q + 127) // 128)
    if qk_head_dim == 128 and bool(
        (base_ctas < _FOUR_WAVE_PREFILL_CTAS_PER_CU * num_cus)
        & ((seq_kv <= 1024) | (base_ctas < num_cus))
    ):
        return 2
    return 4


def _select_waves_per_eu(
    *,
    owner_waves: int,
    decode: bool,
    prefill_ctas,
    seq_kv,
    qk_head_dim: int,
    num_cus: int,
) -> int:
    """Select the occupancy hint independently from the owner geometry."""
    if qk_head_dim != 128 or owner_waves not in (4, 8):
        return 1
    enough_work = seq_kv >= 2048
    if not decode:
        enough_work = enough_work | (
            prefill_ctas >= _FOUR_WAVE_PREFILL_CTAS_PER_CU * num_cus
        )
    return 2 if bool(enough_work) else 1


def forward_plan(
    *,
    batch_size,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q,
    seq_kv,
    qk_head_dim: int,
    v_head_dim: int,
    has_score_mod: bool = False,
    padded_kv: bool = False,
    has_mask_buffers: bool = False,
    kv_index_widths=(),
    single_pass: bool = False,
    target=None,
) -> dict:
    """The forward build choices for ``target`` (``TargetLimits``, default: the
    current device). With runtime sizes the paired schedule unrolls over
    ``paired_kv_blocks``, a power of two bounding the KV blocks and the
    BlockMask list widths ``kv_index_widths``. ``single_pass`` plans the
    head-dim-staged 4-wave pipeline for one launch over all of V."""
    lds_bytes, num_cus = flex_attn_target.resolve_target(target)
    static = all(isinstance(size, int) for size in (batch_size, seq_q, seq_kv))
    decode = bool(seq_q < 128)
    # Pair sparse rows only when enough CTAs and KV work amortize the merge.
    # The staged paired schedule assumes four 32-column V chunks. With a
    # score_mod the paired pipeline spills VGPRs on gfx950.
    paired = (
        not has_score_mod
        and not padded_kv
        and (qk_head_dim, v_head_dim) in ((128, 128), (192, 128))
        and bool(
            (seq_q % 256 == 0)
            & (seq_kv % 128 == 0)
            & (
                batch_size * num_q_heads * (seq_q // 256)
                >= _FOUR_WAVE_PREFILL_CTAS_PER_CU * num_cus
            )
            & (seq_kv // 128 >= (64 if has_mask_buffers else 16))
        )
    )
    paired_kv_blocks = None
    if paired:
        paired_kv_blocks = seq_kv // 128
        if not static:
            paired_kv_blocks = _power_of_two_bound(paired_kv_blocks, *kv_index_widths)
    owner_waves = (
        8
        if paired
        else 4
        if single_pass
        else _select_owner_waves(
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_q=seq_q,
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            num_cus=num_cus,
        )
    )
    wide_head = _is_wide_head(qk_head_dim, v_head_dim)
    split_kv = not wide_head and bool(
        ((num_q_heads // num_kv_heads) * seq_q == 1)
        & (batch_size * num_kv_heads < num_cus)
        & (seq_kv >= 2048)
    )
    waves_per_eu = (
        2
        if paired
        else 1
        if single_pass
        else _select_waves_per_eu(
            owner_waves=owner_waves,
            decode=decode,
            prefill_ctas=batch_size * num_q_heads * ((seq_q + 127) // 128),
            seq_kv=seq_kv,
            qk_head_dim=qk_head_dim,
            num_cus=num_cus,
        )
    )
    if single_pass:
        pipelined_kv = True
    elif not decode:
        # The unpipelined CTA holds its Q tile and one shared 64-row K/V tile.
        unpipelined_lds = _ELEMENT_BYTES * (
            owner_waves * 32 * qk_head_dim + 64 * max(qk_head_dim, v_head_dim)
        )
        pipelined_kv = unpipelined_lds > lds_bytes or bool(seq_kv > 128)
    else:
        pipelined_kv = wide_head or (
            qk_head_dim == 128
            and owner_waves in (2, 4)
            and bool((batch_size * num_kv_heads <= num_cus) & (seq_kv >= 2048))
        )
    # Rows past Sq load zeros through the buffer bound and are never stored;
    # keys past Sk are masked in every KV tile.
    q_ragged = (
        not decode
        and not paired
        and (not isinstance(seq_q, int) or seq_q % (owner_waves * 32) != 0)
    )
    kv_ragged = bool(seq_kv % 128 != 0)
    launch_heads = num_kv_heads if decode else num_q_heads
    # Limit the heads interleaved across query blocks to preserve K/V reuse.
    # A runtime B keeps the group within one batch.
    total_heads = (
        batch_size * launch_heads if isinstance(batch_size, int) else launch_heads
    )
    heads_per_group = max(1, min(total_heads, 64))
    while total_heads % heads_per_group:
        heads_per_group -= 1
    return {
        "decode": decode,
        "paired": bool(paired),
        "paired_kv_blocks": paired_kv_blocks,
        "owner_waves": owner_waves,
        "split_kv": bool(split_kv),
        "waves_per_eu": waves_per_eu,
        "pipelined_kv": pipelined_kv,
        "q_ragged": bool(q_ragged),
        "kv_ragged": kv_ragged,
        "heads_per_group": heads_per_group,
    }


def forward_kv_splits(
    *,
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_q: int,
    seq_kv: int,
    qk_head_dim: int,
    v_head_dim: int,
    requested=True,
    target=None,
) -> int:
    """How many CTAs split each decode CTA's KV lists (``requested`` if an int)."""
    if seq_q >= 128 or requested is False:
        return 1
    num_cus = flex_attn_target.resolve_target(target).num_cus
    kv_blocks = (seq_kv + 127) // 128
    if requested is not True:
        return max(1, min(int(requested), kv_blocks, _MAX_KV_SPLITS))
    owner_waves = _select_owner_waves(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        num_cus=num_cus,
    )
    packed_rows = num_q_heads // num_kv_heads * seq_q
    ctas = batch_size * num_kv_heads * -(-packed_rows // (owner_waves * 32))
    splits = min(
        -(-num_cus // ctas),
        kv_blocks // _SPLIT_KV_MIN_BLOCKS,
        _MAX_KV_SPLITS,
    )
    return max(1, splits)


def backward_plan(
    *,
    batch_size,
    num_heads,
    num_kv_heads,
    sequence_length,
    kv_sequence_length,
    qk_head_dim,
    value_head_dim,
    mask_traversal,
    workspace_element_bytes,
    strides,
    dq_partitions=None,
    captures=0,
    target=None,
):
    """The backward build choices for ``target`` (see ``forward_plan``);
    runtime sizes need ``dq_partitions``.
    ``strides`` are those of Q, K, V, O, dO, dQ, dK and dV. ``dq_accumulates``:
    KV owners add to zeroed dQ slots (always for block lists with a runtime Sk).
    """
    block_list = mask_traversal == MASK_TRAVERSAL_BLOCK_LIST
    key_rows = backward_key_rows(block_list, qk_head_dim, value_head_dim, captures)
    padded_sequence_length = (sequence_length + 127) // 128 * 128
    static_batch = isinstance(batch_size, int)
    if dq_partitions is None:
        dq_partitions = choose_dq_partitions(
            batch_size * num_heads,
            padded_sequence_length,
            key_rows,
            workspace_element_bytes,
            qk_head_dim=qk_head_dim,
            key_sequence_length=kv_sequence_length,
            compute_batch_heads=batch_size * num_kv_heads,
            block_list=block_list,
            target=target,
        )
    if isinstance(kv_sequence_length, int):
        dq_accumulates = dq_partitions < (kv_sequence_length + key_rows - 1) // key_rows
    else:
        dq_accumulates = block_list
    q_ragged = bool(sequence_length % 128 != 0)
    out_stride, grad_output_stride = strides[3:5]
    query_elements = batch_size * num_heads * sequence_length
    linear_delta = query_elements * qk_head_dim * 2 <= 0x7FFFFFFF
    for stride_tuple in (out_stride, grad_output_stride):
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(
                (batch_size, num_heads, sequence_length, value_head_dim), stride_tuple
            )
        )
        linear_delta = linear_delta & (span * 2 <= 0x7FFFFFFF)
        for stride in stride_tuple[:-1]:
            linear_delta = linear_delta & (stride % 8 == 0)
    dense_linear_delta = not block_list and not q_ragged and bool(linear_delta)
    # A runtime B keeps the head groups within one batch.
    group_heads = batch_size * num_heads if static_batch else num_heads
    group_kv_heads = batch_size * num_kv_heads if static_batch else num_kv_heads
    return {
        "q_ragged": q_ragged,
        "kv_ragged": bool(kv_sequence_length % 128 != 0),
        "dq_partitions": int(dq_partitions),
        "dq_accumulates": bool(dq_accumulates),
        "workspace_head_group": dq_workspace_head_group(group_heads),
        "cta_head_group": min(16, group_kv_heads & -group_kv_heads),
        "heads_are_inner": bool(
            functools.reduce(
                operator.and_, (stride[1] < stride[2] for stride in strides)
            )
        ),
        "dense_linear_delta": bool(dense_linear_delta),
    }
