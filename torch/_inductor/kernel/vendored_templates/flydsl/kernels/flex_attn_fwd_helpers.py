# mypy: allow-untyped-defs

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import (
    BFloat16 as bf16,
    const_expr,
    Float32 as f32,
    Int32 as i32,
    Int64 as i64,
    Uint32 as u32,
    Uint64 as u64,
)
from flydsl.expr.typing import Vector as Vec

from .flex_attn_utils import (
    fast_exp2,
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_qk_shared_layout,
    make_value_shared_layout,
    schedule_fwd_pv_pipeline,
    schedule_fwd_qk_pipeline,
    schedule_fwd_softmax_pipeline,
)


unroll = fx.range_constexpr
vector = Vec.from_elements
layout = fx.make_layout
tensor_view = fx.make_view
divide = fx.logical_divide
slice_view = fx.slice
scalar = fx.get_scalar
tensor_copy = fx.copy
schedule_group = fx.rocdl.sched_group_barrier


_LOG2E = 1.4426950408889634
_LN2 = math.log(2.0)
# Keep masked exponentials finite under fast math; empty rows store -inf explicitly.
_NEG_BIG = -1.0e30
# Four-wave CTAs reduce launch count once the 128-row prefill grid remains
# large enough to keep gfx950 occupied.
_FOUR_WAVE_PREFILL_MIN_CTAS = 512


_f32 = f32


def _select_waves_per_eu(
    *, owner_waves: int, enough_prefill_parallelism: bool, seq_kv: int, qk_head_dim: int
) -> int:
    """Select the occupancy hint independently from the owner geometry."""
    if (
        qk_head_dim == 128
        and owner_waves in (4, 8)
        and (seq_kv >= 2048 or enough_prefill_parallelism)
    ):
        return 2
    return 1


def _exp2(value):
    return fast_exp2(_f32(value))


def _maximum(lhs, rhs):
    return lhs.maximumf(rhs)


def _reduce32(values, maximum):
    for level in unroll(5):
        values = [
            _maximum(values[2 * element_index], values[2 * element_index + 1])
            if maximum
            else values[2 * element_index] + values[2 * element_index + 1]
            for element_index in unroll(32 >> (level + 1))
        ]
    return values[0]


def _pin(value):
    raw = value.ir_value()
    return llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True)


@flyc.jit
def make_kv_staging(copies, keys, values, geometry, owners):
    key_copy, lds_copy, transposed_lds_copy, shared_copy, value_copy = copies
    key_copy_coords, key_destinations, key_stages, k_stride, key_tiles, key_view = keys
    (
        value_copy_coords,
        value_destinations,
        value_stages,
        v_stride,
        value_tiles,
        value_view,
    ) = values
    (
        key_loads,
        kv_load_threads,
        kv_tile_rows,
        pack_size,
        qk_head_dim,
        value_head_dim,
        value_loads,
        wave_size,
    ) = geometry
    lane, paired, split_kv, thread_mma, thread_id, wave = owners

    def stage_kv(kv_base, stage, key_tile, first, last):
        view = key_view if key_tile else value_view
        stride = k_stride if key_tile else v_stride
        coordinates = key_copy_coords if key_tile else value_copy_coords
        pointers = key_stages if key_tile else value_stages
        destinations = key_destinations if key_tile else value_destinations

        def pack_coordinates(load_step):
            load_tid = lane if const_expr(split_kv) else thread_id
            linear = i32(load_step * kv_load_threads) + load_tid
            logical = i32(scalar(fx.crd2idx(linear * i32(pack_size), coordinates)))
            row = logical % i32(kv_tile_rows)
            chunk = logical // i32(kv_tile_rows * pack_size)
            return linear, row, chunk

        def destination(load_step):
            # LDS DMA adds the lane offset; keep its base wave-uniform.
            return tensor_view(
                pointers[stage]
                + (i32(load_step * kv_load_threads) + wave * i32(wave_size))
                * i32(pack_size),
                layout(pack_size, 1),
            )

        if const_expr(paired and qk_head_dim > value_head_dim):
            offsets = []
            for load_step in unroll(first, last):
                linear, row, chunk = pack_coordinates(load_step)
                if const_expr(key_tile):
                    offset = row * i32(stride[2])
                else:
                    offset = (kv_base + row) * i32(stride[2])
                offsets.append(offset + chunk * i32(pack_size))
            if const_expr(not key_tile):
                # AMDGPU vector constraints require a supported register tuple size.
                padded_count = max(2, 1 << (len(offsets) - 1).bit_length())
                offsets = offsets + [offsets[0]] * (padded_count - len(offsets))
            packed = vector(offsets, i32)
            if const_expr(not key_tile):
                packed = Vec(_pin(packed))
            for load_step in unroll(first, last):
                source = tensor_view(
                    fx.get_iter(view) + i32(packed[load_step - first]),
                    layout(pack_size, 1),
                )
                copy = lds_copy
                if const_expr(key_tile):
                    copy = copy.set_value("soffset", kv_base * i32(stride[2]))
                tensor_copy(copy, source, destination(load_step))
        else:
            for load_step in unroll(first, last):
                linear, row, chunk = pack_coordinates(load_step)
                source = divide(
                    slice_view(view, (kv_base + row, None)), layout(pack_size, 1)
                )
                if const_expr(paired):
                    tensor_copy(
                        lds_copy,
                        slice_view(source, (None, chunk)),
                        destination(load_step),
                    )
                else:
                    tensor_copy(
                        lds_copy,
                        slice_view(source, (None, chunk)),
                        slice_view(destinations[stage], (None, linear)),
                    )

    def stage_key(kv_base, stage=0, first=0, last=key_loads):
        stage_kv(kv_base, stage, True, first, last)

    def stage_value(kv_base, stage=0):
        stage_kv(kv_base, stage, False, 0, value_loads)

    def load_key(k_step, high_half, stage=0):
        tile = slice_view(key_tiles[stage], (None, None, int(high_half), k_step))
        fragment = thread_mma.make_fragment_A(tile)
        tensor_copy(shared_copy, key_copy.partition_S(tile), key_copy.retile(fragment))
        return fragment

    def load_value(probability_pack, d_chunk, stage=0):
        tile = slice_view(value_tiles[stage], (None, None, d_chunk, probability_pack))
        fragment = thread_mma.make_fragment_A(tile)
        tensor_copy(
            transposed_lds_copy,
            value_copy.partition_S(tile),
            value_copy.retile(fragment),
        )
        return fragment

    return stage_key, stage_value, load_key, load_value


@flyc.jit
def make_tile_processor(geometry, query, mask, compute, mode):
    (
        key_loads,
        kv_tile_rows,
        mma_tile_size,
        output_chunks,
        qk_reduction_steps,
        value_loads,
        wave_size,
    ) = geometry
    accumulator_coords, batch, query_base, query_head, query_pos, query_valid, wave = (
        query
    )
    (
        evaluate_mask,
        supports_mask_intervals,
        load_i32,
        load_mask_group,
        mask_buffers,
        mask_buffer_count,
        mask_output_slot,
        mask_program,
        mask_buffer_strides,
        mask_load_width,
        vector_mask_loads,
    ) = mask
    (
        accumulate_probability,
        load_key,
        load_query,
        mfma,
        reduce_lane_pair,
        stage_key,
        stage_value,
        zero16,
    ) = compute
    paired, pipelined = mode

    def process_tile_body(
        kv_chunk,
        masked,
        tile_output,
        running_peak,
        running_total,
        stage=0,
        active=None,
        exact_max=None,
        prepared_mask=None,
        full=None,
    ):
        kv_base = kv_chunk * i32(kv_tile_rows)
        if const_expr(not pipelined):
            stage_key(kv_base)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.gpu.barrier()

        scores_lo = zero16
        scores_hi = zero16
        for k_step in unroll(qk_reduction_steps):
            query_pack = load_query(k_step)
            key_lo = load_key(k_step, False, stage)
            key_hi = load_key(k_step, True, stage)
            scores_lo = mfma(key_lo, query_pack, scores_lo)
            scores_hi = mfma(key_hi, query_pack, scores_hi)
        schedule_fwd_qk_pipeline(
            reduction_steps=qk_reduction_steps,
            vmem_count=(key_loads if pipelined else 0),
            query_in_registers=pipelined,
        )

        if const_expr(not pipelined):
            # Every wave must finish its K reads before the shared allocation
            # is reused for V.
            fx.gpu.barrier()
            stage_value(kv_base)

        raw_lo = Vec(scores_lo)
        raw_hi = Vec(scores_hi)
        score_values = []
        keep_values = []
        for half in unroll(2):
            raw = raw_lo if half == 0 else raw_hi
            half_mask_groups = []
            if const_expr(prepared_mask is None and bool(vector_mask_loads) and masked):
                for group in unroll(16 // mask_load_width):
                    first_key = (
                        kv_base
                        + i32(32 * half)
                        + i32(scalar(accumulator_coords[group * mask_load_width]))
                    )
                    half_mask_groups.append(load_mask_group(first_key))
            for element in unroll(16):
                key_pos = (
                    kv_base + i32(32 * half) + i32(scalar(accumulator_coords[element]))
                )
                if const_expr(prepared_mask is not None):
                    keep = prepared_mask[half * 16 + element]
                else:
                    keep = query_valid
                    if active is not None:
                        keep = keep & active
                    if const_expr(bool(mask_program)):
                        if masked:
                            cached_mask = {}
                            if const_expr(bool(vector_mask_loads)):
                                mask_group = half_mask_groups[
                                    element // mask_load_width
                                ]
                                cached_mask = {
                                    slot: i32(values[element % mask_load_width])
                                    for slot, values in mask_group.items()
                                }
                            element_keep = evaluate_mask(
                                query_pos, key_pos, cached_mask
                            )
                            if full is not None:
                                element_keep = full | element_keep
                            keep = keep & element_keep

                keep_values.append(keep)
                score_values.append(keep.select(_f32(raw[element]), _f32(_NEG_BIG)))

        local_max = _reduce32(score_values, True)
        if const_expr(pipelined):
            tile_max = reduce_lane_pair(local_max, True)
        else:
            peer_max = _f32(fx.gpu.shuffle_xor(local_max, mma_tile_size, wave_size))
            tile_max = _maximum(local_max, peer_max)
        exact_max = _maximum(exact_max, tile_max)
        if const_expr(pipelined):
            rescale = u64(
                fx.rocdl.ballot(
                    u64.ir_type, (tile_max > running_peak + _f32(8.0)).ir_value()
                )
            ) != u64(0)
            new_max = rescale.select(_maximum(running_peak, tile_max), running_peak)
            correction = _f32(1.0)
            for output_chunk in unroll(output_chunks):
                tile_output[output_chunk] = Vec(_pin(Vec(tile_output[output_chunk])))
            if rescale:
                correction = _exp2(running_peak - new_max)
                correction_vec = vector([correction], f32).broadcast_to(16)
                for output_chunk in unroll(output_chunks):
                    tile_output[output_chunk] = (
                        Vec(tile_output[output_chunk]) * correction_vec
                    )
            for output_chunk in unroll(output_chunks):
                tile_output[output_chunk] = Vec(_pin(Vec(tile_output[output_chunk])))

        else:
            new_max = _maximum(running_peak, tile_max)
            correction = _exp2(running_peak - new_max)

            correction_vec = vector([correction], f32).broadcast_to(16)
            for d_chunk in unroll(output_chunks):
                tile_output[d_chunk] = Vec(tile_output[d_chunk]) * correction_vec

        stream_probabilities = (
            not paired and pipelined and mask_buffer_count > 0 and masked
        )
        shifted_scores = []
        for pack in unroll(4):
            shifted = vector(
                [
                    score_values[pack * 8 + element_index] - new_max
                    for element_index in unroll(8)
                ],
                f32,
            )
            if const_expr(pipelined and not masked):
                shifted = Vec(_pin(shifted))
            shifted_scores.extend(
                [shifted[element_index] for element_index in unroll(8)]
            )
        sum_probs = []
        prob_packs = []
        for pack in unroll(4):
            pack_probs = []
            for pack_element in unroll(8):
                element = pack * 8 + pack_element
                probability = keep_values[element].select(
                    _exp2(shifted_scores[element]), _f32(0.0)
                )
                pack_probs.append(probability)
                sum_probs.append(probability)
            pack_values = vector(pack_probs, f32).to(bf16)
            if const_expr(stream_probabilities):
                tile_output = accumulate_probability(
                    pack_values, pack, tile_output, stage
                )
            else:
                prob_packs.append(pack_values)
        local_sum = _reduce32(sum_probs, False)
        if const_expr(not stream_probabilities):
            schedule_fwd_softmax_pipeline(vmem_count=value_loads)
        if const_expr(pipelined):
            tile_sum = reduce_lane_pair(local_sum, False)
        else:
            peer_sum = _f32(fx.gpu.shuffle_xor(local_sum, mma_tile_size, wave_size))
            tile_sum = local_sum + peer_sum
        running_total = running_total * correction + tile_sum
        running_peak = new_max

        if const_expr(not pipelined):
            # V writes were issued before the register-only softmax.
            # Synchronize only when the LDS data is actually consumed.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.gpu.barrier()
        if const_expr(not stream_probabilities):
            for probability_pack in unroll(4):
                tile_output = accumulate_probability(
                    prob_packs[probability_pack], probability_pack, tile_output, stage
                )
            schedule_fwd_pv_pipeline(output_chunks=output_chunks)

        if const_expr(not pipelined):
            # Protect V from the next tile's K staging.
            fx.gpu.barrier()
        return tile_output, running_peak, running_total, exact_max

    def wave_mask_bounds(kv_chunk):
        if const_expr(mask_buffer_count):
            query_lo = query_pos
            query_hi = query_pos
        else:
            query_lo = query_base + wave * i32(mma_tile_size)
            query_hi = query_lo + i32(mma_tile_size - 1)
        key_lo = kv_chunk * i32(kv_tile_rows)
        key_hi = key_lo + i32(kv_tile_rows - 1)
        intervals = [
            (i32(batch), i32(batch)),
            (i32(query_head), i32(query_head)),
            (query_lo, query_hi),
            (key_lo, key_hi),
        ]
        for instruction in mask_program:
            opcode = instruction[0]
            if opcode == "const_i32":
                bound_value = i32(instruction[1])
                intervals.append((bound_value, bound_value))
            elif opcode == "const_bool":
                bound_value = i32(1 if instruction[1] else 0) == i32(1)
                intervals.append((bound_value, bound_value))
            elif opcode == "load_i32":
                buffer_index, index_ids = instruction[1:]
                offset = i32(0)
                for dimension, index in enumerate(index_ids):
                    offset = offset + intervals[index][0] * i32(
                        mask_buffer_strides[buffer_index][dimension]
                    )
                loaded_bound = load_i32(mask_buffers[buffer_index], offset)
                intervals.append((loaded_bound, loaded_bound))
            else:
                lhs_lo, lhs_hi = intervals[instruction[1]]
                if opcode == "not":
                    intervals.append((~lhs_hi, ~lhs_lo))
                else:
                    rhs_lo, rhs_hi = intervals[instruction[2]]
                    if opcode == "ge":
                        intervals.append((lhs_lo >= rhs_hi, lhs_hi >= rhs_lo))
                    elif opcode == "gt":
                        intervals.append((lhs_lo > rhs_hi, lhs_hi > rhs_lo))
                    elif opcode == "le":
                        intervals.append((lhs_hi <= rhs_lo, lhs_lo <= rhs_hi))
                    elif opcode == "lt":
                        intervals.append((lhs_hi < rhs_lo, lhs_lo < rhs_hi))
                    elif opcode == "eq":
                        certain = (
                            (lhs_lo == lhs_hi) & (rhs_lo == rhs_hi) & (lhs_lo == rhs_lo)
                        )
                        possible = (lhs_lo <= rhs_hi) & (rhs_lo <= lhs_hi)
                        intervals.append((certain, possible))
                    elif opcode == "ne":
                        certain = (lhs_hi < rhs_lo) | (rhs_hi < lhs_lo)
                        possible = (
                            (lhs_lo != lhs_hi) | (rhs_lo != rhs_hi) | (lhs_lo != rhs_lo)
                        )
                        intervals.append((certain, possible))
                    elif opcode == "and":
                        intervals.append((lhs_lo & rhs_lo, lhs_hi & rhs_hi))
                    elif opcode == "or":
                        intervals.append((lhs_lo | rhs_lo, lhs_hi | rhs_hi))
        return intervals[mask_output_slot]

    def process_tile(
        kv_chunk,
        masked,
        tile_output,
        running_peak,
        running_total,
        stage=0,
        active=None,
        exact_max=None,
        full=None,
    ):
        if const_expr(exact_max is None):
            exact_max = running_peak
        if const_expr(paired and supports_mask_intervals and masked):
            certain, possible = wave_mask_bounds(kv_chunk)
            if full is not None:
                possible = possible | full
            if active is not None:
                possible = possible & active
            if const_expr(mask_buffer_count):
                possible = u64(
                    fx.rocdl.ballot(u64.ir_type, possible.ir_value())
                ) != u64(0)
            if possible:
                (tile_output, running_peak, running_total, exact_max) = (
                    process_tile_body(
                        kv_chunk,
                        True,
                        tile_output,
                        running_peak,
                        running_total,
                        stage,
                        active,
                        exact_max,
                        full=full,
                    )
                )
            return (tile_output, running_peak, running_total, exact_max)
        elif const_expr(
            not paired and pipelined and masked and bool(vector_mask_loads)
        ):
            prepared_mask = []
            kv_base = kv_chunk * i32(kv_tile_rows)
            for half in unroll(2):
                mask_groups = []
                for group in unroll(16 // mask_load_width):
                    first_key = (
                        kv_base
                        + i32(32 * half)
                        + i32(scalar(accumulator_coords[group * mask_load_width]))
                    )
                    mask_groups.append(load_mask_group(first_key))
                for element in unroll(16):
                    key_pos = (
                        kv_base
                        + i32(32 * half)
                        + i32(scalar(accumulator_coords[element]))
                    )
                    cached_mask = {
                        slot: i32(values[element % mask_load_width])
                        for slot, values in mask_groups[
                            element // mask_load_width
                        ].items()
                    }
                    keep = query_valid & evaluate_mask(query_pos, key_pos, cached_mask)
                    if active is not None:
                        keep = keep & active
                    prepared_mask.append(keep)
            levels = prepared_mask
            for level in unroll(5):
                levels = [
                    levels[2 * pair] | levels[2 * pair + 1]
                    for pair in unroll(32 >> (level + 1))
                ]
            any_work = u64(fx.rocdl.ballot(u64.ir_type, levels[0].ir_value())) != u64(0)
            if any_work:
                (tile_output, running_peak, running_total, exact_max) = (
                    process_tile_body(
                        kv_chunk,
                        masked,
                        tile_output,
                        running_peak,
                        running_total,
                        stage,
                        active,
                        exact_max,
                        prepared_mask=prepared_mask,
                    )
                )
            return (tile_output, running_peak, running_total, exact_max)
        return process_tile_body(
            kv_chunk,
            masked,
            tile_output,
            running_peak,
            running_total,
            stage,
            active,
            exact_max,
            full=full,
        )

    return process_tile
