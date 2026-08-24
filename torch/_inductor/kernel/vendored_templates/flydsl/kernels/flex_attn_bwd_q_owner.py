"""Parameterised dQ owner pipeline for FlexAttention backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_utils import (
    make_global_view,
    make_mask_buffers,
    make_mask_evaluator,
    make_mfma32_ops,
    schedule_pack0_pipeline,
    schedule_pack1_pipeline,
    schedule_score_pipeline,
    schedule_update_tail,
    scheduled_workgroup_barrier,
)

_LOG2E = 1.4426950408889634
_NEG_BIG = -1e30


def make_dq_mfma32_body(context, f32, exp2):
    CAUSAL_MASK = context["CAUSAL_MASK"]
    DENSE_MASK = context["DENSE_MASK"]
    WINDOW_MASK = context["WINDOW_MASK"]
    WINDOW_SIZE = context["WINDOW_SIZE"]
    DO_STRIDE = context["DO_STRIDE"]
    DQ32_BM = context["DQ32_BM"]
    DQ32_BN = context["DQ32_BN"]
    DQ32_CAUSAL_TILES = context["DQ32_CAUSAL_TILES"]
    DQ32_CE = context["DQ32_CE"]
    DQ32_K_LOAD_IT = context["DQ32_K_LOAD_IT"]
    DQ32_V_LOAD_IT = context["DQ32_V_LOAD_IT"]
    DQ32_NC = context["DQ32_NC"]
    DQ32_NT = context["DQ32_NT"]
    DQK = context["DQK"]
    DQ_STRIDE = context["DQ_STRIDE"]
    DV = context["DV"]
    H = context["H"]
    K_STRIDE = context["K_STRIDE"]
    LSE_IN_LOG2 = context["LSE_IN_LOG2"]
    LSTRIDE = context["LSTRIDE"]
    MASK_BUFFER_COUNT = context["MASK_BUFFER_COUNT"]
    MASK_BUFFER_SIZES = context["MASK_BUFFER_SIZES"]
    MASK_BUFFER_STRIDES = context["MASK_BUFFER_STRIDES"]
    MASK_PROGRAM = context["MASK_PROGRAM"]
    MASK_PROGRAM_OUTPUT = context["MASK_PROGRAM_OUTPUT"]
    MC = context["MC"]
    NC = context["NC"]
    Q_STRIDE = context["Q_STRIDE"]
    S = context["S"]
    SC2 = context["SC2"]
    SCALE = context["SCALE"]
    VPT = context["VPT"]
    V_STRIDE = context["V_STRIDE"]
    gview = make_global_view
    _f32 = f32
    _exp2 = exp2

    @flyc.jit
    def _emit_dq_mfma32_body(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        CP: fx.Tensor,
        IP: fx.Tensor,
        CF: fx.Tensor,
        IF: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
        mc: fx.Int32,
        bh: fx.Int32,
        ppersistent,
        pshort,
    ):
        """dQ owner with asymmetric operand-native LDS staging.

        Q and dO are loaded directly into persistent MFMA registers. Each KV
        tile keeps K in two alternating LDS slots because both score and dQ
        consume it. V is score-only and reuses one LDS slot after score/dP.
        """
        tid = fx.Int32(fx.thread_idx.x)
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        lane_row = lane % fx.Int32(32)
        lane_half = lane // fx.Int32(32)
        qbase = mc * fx.Int32(DQ32_BM)
        qlocal = fx.Int32(32) * wave + lane_row
        qi = qbase + qlocal
        loff = bh * fx.Int32(LSTRIDE)
        qoff = batch * fx.Int32(Q_STRIDE[0]) + head * fx.Int32(Q_STRIDE[1])
        koff = batch * fx.Int32(K_STRIDE[0]) + head * fx.Int32(K_STRIDE[1])
        voff = batch * fx.Int32(V_STRIDE[0]) + head * fx.Int32(V_STRIDE[1])
        dooff = batch * fx.Int32(DO_STRIDE[0]) + head * fx.Int32(DO_STRIDE[1])
        dqoff = batch * fx.Int32(DQ_STRIDE[0]) + head * fx.Int32(DQ_STRIDE[1])
        gQ = gview(Q, qoff, (S, DQK), (Q_STRIDE[2], Q_STRIDE[3]))
        gK = gview(K, koff, (S, DQK), (K_STRIDE[2], K_STRIDE[3]))
        gV = gview(V, voff, (S, DV), (V_STRIDE[2], V_STRIDE[3]))
        gDO = gview(DO, dooff, (S, DV), (DO_STRIDE[2], DO_STRIDE[3]))
        gDQ = gview(DQ, dqoff, (S, DQK), (DQ_STRIDE[2], DQ_STRIDE[3]))
        gLSE = gview(LSE, loff, S, 1)
        gDEL = gview(DELTA, loff, S, 1)
        gCP = gview(CP, bh * fx.Int32(MC), MC, 1)
        gCF = gview(CF, bh * fx.Int32(MC), MC, 1)
        gIP = gview(IP, bh * fx.Int32(MC * NC), MC * NC, 1)
        gIF = gview(IF, bh * fx.Int32(MC * NC), MC * NC, 1)
        MaskBuffers = make_mask_buffers(
            gview,
            MASK_BUFFER_COUNT,
            MASK_BUFFER_SIZES,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        )
        (
            dma128,
            tr16,
            o64,
            atom,
            gload_f32,
            gload_i32,
            load_global_pack,
            make_fragment,
            mfma,
            b_operand_column,
            dma_destination,
        ) = make_mfma32_ops(WINDOW_MASK, VPT)
        evaluate_mask = make_mask_evaluator(
            MASK_PROGRAM,
            MASK_PROGRAM_OUTPUT,
            MASK_BUFFER_STRIDES,
            MaskBuffers,
            gload_i32,
            batch,
            head,
        )

        def swizzled_32x32_offset(row, column):
            return row * fx.Int32(32) + (
                column ^ (row & fx.Int32(8)) << fx.Int32(1) ^ (row & fx.Int32(16)) >> fx.Int32(1)
            )

        def stage_a_operand(
            view,
            destination,
            kchunk,
            stage: fx.Constexpr[int],
            dim: fx.Constexpr[int],
            load_iterations: fx.Constexpr[int],
        ):
            kvbase = kchunk * fx.Int32(DQ32_BN)
            stage_offset = fx.Int32(stage * DQ32_BN * dim)
            for load_step in fx.range_constexpr(load_iterations):
                linear = fx.Int32(load_step * DQ32_NT) + tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(32 * 32)
                within_subtile = lds_offset % fx.Int32(32 * 32)
                row_block = subtile // fx.Int32(dim // 32)
                d_subtile = subtile % fx.Int32(dim // 32)
                row = row_block * fx.Int32(32) + within_subtile // fx.Int32(32)
                swizzled_column = within_subtile % fx.Int32(32)
                column = d_subtile * fx.Int32(32) + (
                    swizzled_column ^ (row & fx.Int32(8)) << fx.Int32(1) ^ (row & fx.Int32(16)) >> fx.Int32(1)
                )
                source = fx.logical_divide(fx.slice(view, (kvbase + row, None)), fx.make_layout(VPT, 1))
                fx.copy(
                    dma128,
                    fx.slice(source, (None, column // fx.Int32(VPT))),
                    dma_destination(destination, stage_offset + lds_offset),
                )

        def stage_b_operand(view, destination, kchunk, stage: fx.Constexpr[int]):
            kvbase = kchunk * fx.Int32(DQ32_BN)
            stage_offset = fx.Int32(stage * DQ32_BN * DQK)
            for load_step in fx.range_constexpr(DQ32_K_LOAD_IT):
                linear = fx.Int32(load_step * DQ32_NT) + tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(8 * 32)
                within_subtile = lds_offset % fx.Int32(8 * 32)
                row_in_group = within_subtile // fx.Int32(32)
                column_in_subtile = within_subtile % fx.Int32(32)
                row_group = subtile // fx.Int32(DQK // 32)
                d_subtile = subtile % fx.Int32(DQK // 32)
                row = row_group * fx.Int32(8) + row_in_group
                column = d_subtile * fx.Int32(32) + b_operand_column(row_group, column_in_subtile)
                source = fx.logical_divide(fx.slice(view, (kvbase + row, None)), fx.make_layout(VPT, 1))
                fx.copy(
                    dma128,
                    fx.slice(source, (None, column // fx.Int32(VPT))),
                    dma_destination(destination, stage_offset + lds_offset),
                )

        def stage_persistent(kchunk, stage: fx.Constexpr[int]):
            stage_b_operand(gK, ppersistent, kchunk, stage)

        def stage_short(kchunk):
            stage_a_operand(gV, pshort, kchunk, 0, DV, DQ32_V_LOAD_IT)

        def load_a_pack(base, stage: fx.Constexpr[int], row_block, k_step, dim: fx.Constexpr[int]):
            stage_offset = fx.Int32(stage * DQ32_BN * dim)
            d_subtile = fx.Int32(k_step) // fx.Int32(2)
            d_half = fx.Int32(k_step) % fx.Int32(2)
            column = d_half * fx.Int32(16) + lane_half * fx.Int32(VPT)
            element_offset = (
                stage_offset
                + (row_block * fx.Int32(dim // 32) + d_subtile) * fx.Int32(32 * 32)
                + swizzled_32x32_offset(lane_row, column)
            )
            source = fx.make_view(fx.add_offset(base, fx.make_int_tuple(element_offset)), fx.make_layout(VPT, 1))
            return source.load()

        def load_a_pack_from_b_layout(base, stage: fx.Constexpr[int], row_block, k_step):
            stage_offset = fx.Int32(stage * DQ32_BN * DQK)
            d_subtile = fx.Int32(k_step) // fx.Int32(2)
            d_half = fx.Int32(k_step) % fx.Int32(2)
            row_group = row_block * fx.Int32(4) + lane_row // fx.Int32(8)
            row_in_group = lane_row % fx.Int32(8)
            column = d_half * fx.Int32(16) + lane_half * fx.Int32(VPT)
            element_offset = (
                stage_offset
                + (row_group * fx.Int32(DQK // 32) + d_subtile) * fx.Int32(8 * 32)
                + row_in_group * fx.Int32(32)
                + b_operand_column(row_group, column)
            )
            source = fx.make_view(fx.add_offset(base, fx.make_int_tuple(element_offset)), fx.make_layout(VPT, 1))
            return source.load()

        def load_b_pack(base, stage: fx.Constexpr[int], row_block, k_step, d_chunk):
            stage_offset = fx.Int32(stage * DQ32_BN * DQK)
            row_offset = lane % fx.Int32(16) // fx.Int32(4) + lane_half * fx.Int32(4)
            column_offset = lane % fx.Int32(4) * fx.Int32(4) + lane % fx.Int32(32) // fx.Int32(16) * fx.Int32(16)
            lane_offset = row_offset * fx.Int32(32) + column_offset
            halves = []
            for half in fx.range_constexpr(2):
                subtile = (row_block * fx.Int32(4) + fx.Int32(k_step) * fx.Int32(2) + fx.Int32(half)) * fx.Int32(
                    DQK // 32
                ) + fx.Int32(d_chunk)
                source = fx.make_view(
                    fx.add_offset(
                        base,
                        fx.make_int_tuple(
                            stage_offset
                            + subtile * fx.Int32(8 * 32)
                            + lane_offset
                            - column_offset
                            + b_operand_column(row_block * 4 + k_step * 2 + half, column_offset)
                        ),
                    ),
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_rmem_tensor(4, fx.BFloat16)
                fx.copy(tr16, source, fragment)
                halves.append(fx.Vector(fragment.load()))
            return halves[0].shuffle(halves[1], list(range(8))).ir_value()

        q_packs = []
        for k_step in fx.range_constexpr(DQK // 16):
            column = fx.Int32(k_step * 16) + lane_half * fx.Int32(VPT)
            q_packs.append(load_global_pack(gQ, qi, column))
        do_packs = []
        for k_step in fx.range_constexpr(DV // 16):
            column = fx.Int32(k_step * 16) + lane_half * fx.Int32(VPT)
            do_packs.append(load_global_pack(gDO, qi, column))
        lse = gload_f32(gLSE, qi)
        delta = gload_f32(gDEL, qi)
        lse_log2 = lse if LSE_IN_LOG2 else lse * _f32(_LOG2E)
        lse_log2 = (lse < _f32(_NEG_BIG)).select(_f32(0.0), lse_log2)
        zero16 = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        dq_acc = [fx.make_rmem_tensor(16, fx.Float32) for _ in fx.range_constexpr(DQK // 32)]
        for d_chunk in fx.range_constexpr(DQK // 32):
            dq_acc[d_chunk].fill(0)

        def process_tile(
            kchunk, masked, stage: fx.Constexpr[int], next_kchunk, overlap_next, handoff: fx.Constexpr[bool]
        ):
            for reduction_sub in fx.range_constexpr(DQ32_BN // 32):
                score = zero16
                dp = zero16
                for k_step in fx.range_constexpr(DQK // 16):
                    score = mfma(
                        load_a_pack_from_b_layout(ppersistent, stage, reduction_sub, k_step), q_packs[k_step], score
                    )
                for k_step in fx.range_constexpr(DV // 16):
                    dp = mfma(load_a_pack(pshort, 0, reduction_sub, k_step, DV), do_packs[k_step], dp)
                if const_expr(not WINDOW_MASK):
                    schedule_score_pipeline(
                        mfma_count=(DQK + DV) // 16,
                        dsrd_count=(DQK + DV) // 16,
                        vmem_count=DQ32_K_LOAD_IT if overlap_next and reduction_sub == 0 else 0,
                    )
                recycle_short = overlap_next and reduction_sub == DQ32_BN // 32 - 1
                if const_expr(recycle_short):
                    scheduled_workgroup_barrier()
                    stage_short(next_kchunk)
                kvbase = kchunk * fx.Int32(DQ32_BN) + fx.Int32(reduction_sub * 32)
                key_lane_base = fx.Int32(4) * lane_half
                score_values = fx.Vector(score)
                dp_values = fx.Vector(dp)
                first_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = element
                    key_local = key_lane_base + fx.Int32(DQ32_CE[score_index])
                    key_pos = kvbase + key_local
                    probability = _exp2(_f32(score_values[score_index]) * _f32(SC2) - lse_log2)
                    if const_expr(WINDOW_MASK):
                        keep = (qi >= key_pos) & (qi - key_pos < fx.Int32(WINDOW_SIZE))
                        probability = keep.select(probability, _f32(0.0))
                    elif const_expr(CAUSAL_MASK) and masked:
                        probability = (qi >= key_pos).select(probability, _f32(0.0))
                    elif masked:
                        probability = evaluate_mask(qi, key_pos).select(probability, _f32(0.0))
                    first_ds_values.append(probability * (_f32(dp_values[score_index]) - delta))
                first_ds_fragment = make_fragment(
                    fx.Vector.from_elements(first_ds_values, fx.Float32).to(fx.BFloat16).ir_value(), 8, fx.BFloat16
                )
                first_update_fragments = []
                for d_chunk in fx.range_constexpr(DQK // 32):
                    first_update_fragments.append(
                        make_fragment(load_b_pack(ppersistent, stage, reduction_sub, 0, d_chunk), 8, fx.BFloat16)
                    )
                schedule_pack0_pipeline(
                    vmem_count=DQ32_K_LOAD_IT if recycle_short else 0,
                    exp_count=8,
                    dsrd_count=4 * (DQK // 32),
                )
                second_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = 8 + element
                    key_local = key_lane_base + fx.Int32(DQ32_CE[score_index])
                    key_pos = kvbase + key_local
                    probability = _exp2(_f32(score_values[score_index]) * _f32(SC2) - lse_log2)
                    if const_expr(WINDOW_MASK):
                        keep = (qi >= key_pos) & (qi - key_pos < fx.Int32(WINDOW_SIZE))
                        probability = keep.select(probability, _f32(0.0))
                    elif const_expr(CAUSAL_MASK) and masked:
                        probability = (qi >= key_pos).select(probability, _f32(0.0))
                    elif masked:
                        probability = evaluate_mask(qi, key_pos).select(probability, _f32(0.0))
                    second_ds_values.append(probability * (_f32(dp_values[score_index]) - delta))
                second_ds_fragment = make_fragment(
                    fx.Vector.from_elements(second_ds_values, fx.Float32).to(fx.BFloat16).ir_value(), 8, fx.BFloat16
                )
                second_update_fragments = []
                for d_chunk in fx.range_constexpr(min(2, DQK // 32)):
                    second_update_fragments.append(
                        make_fragment(load_b_pack(ppersistent, stage, reduction_sub, 1, d_chunk), 8, fx.BFloat16)
                    )
                for d_chunk in fx.range_constexpr(DQK // 32):
                    fx.gemm(atom, dq_acc[d_chunk], first_update_fragments[d_chunk], first_ds_fragment, dq_acc[d_chunk])
                schedule_pack1_pipeline(
                    mfma_count=DQK // 32,
                    exp_count=8,
                    dsrd_count=2 * (DQK // 32),
                )
                for d_chunk in fx.range_constexpr(DQK // 32):
                    fx.gemm(
                        atom, dq_acc[d_chunk], second_update_fragments[d_chunk], second_ds_fragment, dq_acc[d_chunk]
                    )
                    if const_expr(d_chunk + 2 < DQK // 32):
                        second_update_fragments.append(
                            make_fragment(
                                load_b_pack(ppersistent, stage, reduction_sub, 1, d_chunk + 2), 8, fx.BFloat16
                            )
                        )
                schedule_update_tail(mfma_count=DQK // 32, dsrd_count=2 * (DQK // 32))
            if const_expr(handoff):
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

        direct_mask = DENSE_MASK or CAUSAL_MASK or WINDOW_MASK
        if const_expr(direct_mask):
            if const_expr(DENSE_MASK):
                first_tile = fx.Int32(0)
                tile_count = fx.Int32(DQ32_NC)
            elif const_expr(WINDOW_MASK):
                qbase_window = mc * fx.Int32(DQ32_BM)
                first_tile = (qbase_window >= fx.Int32(WINDOW_SIZE)).select(
                    (qbase_window - fx.Int32(WINDOW_SIZE)) // fx.Int32(DQ32_BN), fx.Int32(0)
                )
                tile_count = mc * fx.Int32(DQ32_CAUSAL_TILES) + fx.Int32(DQ32_CAUSAL_TILES) - first_tile
            else:
                first_tile = fx.Int32(0)
                tile_count = mc * fx.Int32(DQ32_CAUSAL_TILES) + fx.Int32(DQ32_CAUSAL_TILES)

            def direct_masked(tile):
                if const_expr(DENSE_MASK):
                    return False
                if const_expr(WINDOW_MASK):
                    return True
                return tile >= mc * fx.Int32(DQ32_CAUSAL_TILES)

            stage_persistent(first_tile, 0)
            stage_short(first_tile)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            stage_persistent(first_tile + fx.Int32(1), 1)
            process_tile(first_tile, direct_masked(first_tile), 0, first_tile + fx.Int32(1), True, True)
            remaining_tiles = tile_count - fx.Int32(1)
            double_buffered_tiles = remaining_tiles - remaining_tiles % fx.Int32(2)
            for tile_index in range(fx.Int32(0), double_buffered_tiles, fx.Int32(2)):
                local_tile = fx.Int32(tile_index) + fx.Int32(1)
                current_tile = first_tile + local_tile
                stage_persistent(current_tile + fx.Int32(1), 0)
                process_tile(current_tile, direct_masked(current_tile), 1, current_tile + fx.Int32(1), True, True)
                next_pair_tile = local_tile + fx.Int32(2)
                if next_pair_tile < tile_count:
                    stage_persistent(first_tile + next_pair_tile, 1)
                process_tile(
                    current_tile + fx.Int32(1),
                    direct_masked(current_tile + fx.Int32(1)),
                    0,
                    first_tile + next_pair_tile,
                    True,
                    True,
                )
            if double_buffered_tiles < remaining_tiles:
                current_tile = first_tile + double_buffered_tiles + fx.Int32(1)
                process_tile(current_tile, direct_masked(current_tile), 1, current_tile, False, False)
        else:
            rowq = mc * fx.Int32(NC)
            cntp = gload_i32(gCP, mc)
            cntf = gload_i32(gCF, mc)
            for ti in range(fx.Int32(0), cntp, fx.Int32(1)):
                kchunk = gload_i32(gIP, rowq + fx.Int32(ti))
                stage_persistent(kchunk, 0)
                stage_short(kchunk)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                process_tile(kchunk, True, 0, kchunk, False, False)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            for ti in range(fx.Int32(0), cntf, fx.Int32(1)):
                kchunk = gload_i32(gIF, rowq + fx.Int32(ti))
                stage_persistent(kchunk, 0)
                stage_short(kchunk)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                process_tile(kchunk, False, 0, kchunk, False, False)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
        output_row = fx.logical_divide(fx.slice(gDQ, (qi, None)), fx.make_layout(4, 1))
        output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
        output_column_base = fx.Int32(4) * lane_half
        scale_vector = fx.Vector.filled(16, SCALE, fx.Float32)
        for d_chunk in fx.range_constexpr(DQK // 32):
            values = fx.Vector(dq_acc[d_chunk].load()) * scale_vector
            values_bf16 = values.to(fx.BFloat16)
            for pack in fx.range_constexpr(4):
                output = fx.Vector.from_elements(
                    [values_bf16[pack * 4 + element] for element in fx.range_constexpr(4)], fx.BFloat16
                )
                output_fragment.store(output.ir_value())
                column = fx.Int32(d_chunk * 32) + output_column_base + fx.Int32(pack * 8)
                fx.copy(o64, output_fragment, fx.slice(output_row, (None, column // fx.Int32(4))))

    return _emit_dq_mfma32_body
