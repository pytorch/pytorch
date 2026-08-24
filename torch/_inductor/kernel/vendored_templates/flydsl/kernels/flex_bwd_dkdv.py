"""Parameterised paired dK/dV owner for FlexAttention backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_utils import make_global_view, make_shared_view
from .flex_bwd_utils import (
    make_mask_buffers,
    make_mask_evaluator,
    make_mfma32_ops,
    schedule_fence,
    schedule_pack0_pipeline,
    schedule_pack1_pipeline,
    schedule_score_pipeline,
    schedule_update_tail,
)

_LOG2E = 1.4426950408889634
_NEG_BIG = -1e30
_MASK_NONE = 0
_MASK_ALL = 1
_MASK_CAUSAL_BOUNDARY = 2


def make_dkdv_mfma32_body(context, f32, exp2):
    CAUSAL_MASK = context["CAUSAL_MASK"]
    DENSE_MASK = context["DENSE_MASK"]
    WINDOW_MASK = context["WINDOW_MASK"]
    WINDOW_SIZE = context["WINDOW_SIZE"]
    PAIR_BM = context["PAIR_BM"]
    PAIR_BN = context["PAIR_BN"]
    PAIR_CAUSAL_TILES = context["PAIR_CAUSAL_TILES"]
    PAIR_DO_LOAD_IT = context["PAIR_DO_LOAD_IT"]
    PAIR_KEY_SPLIT = context["PAIR_KEY_SPLIT"]
    PAIR_LIST_Q_SPLIT = context["PAIR_LIST_Q_SPLIT"]
    PAIR_LOGICAL_WAVES = context["PAIR_LOGICAL_WAVES"]
    PAIR_MC = context["PAIR_MC"]
    PAIR_Q_LOAD_IT = context["PAIR_Q_LOAD_IT"]
    PAIR_WINDOW_TILES = context["PAIR_WINDOW_TILES"]
    DK_STRIDE = context["DK_STRIDE"]
    DO_STRIDE = context["DO_STRIDE"]
    DQ32_CE = context["DQ32_CE"]
    DQK = context["DQK"]
    DV = context["DV"]
    DV_STRIDE = context["DV_STRIDE"]
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
    COMPUTE_LD_ELEMS = context["COMPUTE_LD_ELEMS"]
    Q_STRIDE = context["Q_STRIDE"]
    S = context["S"]
    SC2 = context["SC2"]
    SCALE = context["SCALE"]
    VPT = context["VPT"]
    V_STRIDE = context["V_STRIDE"]
    _f32 = f32
    _exp2 = exp2

    @flyc.jit
    def _emit_dkdv_mfma32_body(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DVALUE: fx.Tensor,
        CP: fx.Tensor,
        IP: fx.Tensor,
        CF: fx.Tensor,
        IF: fx.Tensor,
        MaskBuffer0: fx.Tensor,
        MaskBuffer1: fx.Tensor,
        MaskBuffer2: fx.Tensor,
        MaskBuffer3: fx.Tensor,
        nc: fx.Int32,
        bh: fx.Int32,
        pqdo,
        pprob,
        pld,
    ):
        """Compute dK and dV with paired producer and consumer waves.

        Producers calculate P/dS/dK; consumers reuse P for dV while loading the
        next tile. DQK and DV have independent compile-time chunks and layouts.
        """
        tid = fx.Int32(fx.thread_idx.x)
        batch = bh // fx.Int32(H)
        head = bh % fx.Int32(H)
        lane = tid % fx.Int32(64)
        wave = tid // fx.Int32(64)
        logical_wave = wave % fx.Int32(PAIR_LOGICAL_WAVES)
        producer_wave = wave < fx.Int32(PAIR_LOGICAL_WAVES)
        lane_row = lane % fx.Int32(32)
        lane_half = lane // fx.Int32(32)
        kvbase = nc * fx.Int32(PAIR_BN)
        key_pos = kvbase + logical_wave * fx.Int32(32) + lane_row
        loff = bh * fx.Int32(LSTRIDE)
        qoff = batch * fx.Int32(Q_STRIDE[0]) + head * fx.Int32(Q_STRIDE[1])
        koff = batch * fx.Int32(K_STRIDE[0]) + head * fx.Int32(K_STRIDE[1])
        voff = batch * fx.Int32(V_STRIDE[0]) + head * fx.Int32(V_STRIDE[1])
        dooff = batch * fx.Int32(DO_STRIDE[0]) + head * fx.Int32(DO_STRIDE[1])
        dkoff = batch * fx.Int32(DK_STRIDE[0]) + head * fx.Int32(DK_STRIDE[1])
        dvoff = batch * fx.Int32(DV_STRIDE[0]) + head * fx.Int32(DV_STRIDE[1])
        gQ = make_global_view(Q, qoff, (S, DQK), (Q_STRIDE[2], Q_STRIDE[3]))
        gK = make_global_view(K, koff, (S, DQK), (K_STRIDE[2], K_STRIDE[3]))
        gV = make_global_view(V, voff, (S, DV), (V_STRIDE[2], V_STRIDE[3]))
        gDO = make_global_view(DO, dooff, (S, DV), (DO_STRIDE[2], DO_STRIDE[3]))
        gDK = make_global_view(DK, dkoff, (S, DQK), (DK_STRIDE[2], DK_STRIDE[3]))
        gDV = make_global_view(DVALUE, dvoff, (S, DV), (DV_STRIDE[2], DV_STRIDE[3]))
        gLSE = make_global_view(LSE, loff, S, 1)
        gDEL = make_global_view(DELTA, loff, S, 1)
        gLSE4 = fx.logical_divide(gLSE, fx.make_layout(4, 1))
        gDEL4 = fx.logical_divide(gDEL, fx.make_layout(4, 1))
        gCP = make_global_view(CP, bh * fx.Int32(NC), NC, 1)
        gCF = make_global_view(CF, bh * fx.Int32(NC), NC, 1)
        gIP = make_global_view(IP, bh * fx.Int32(NC * MC), NC * MC, 1)
        gIF = make_global_view(IF, bh * fx.Int32(NC * MC), NC * MC, 1)
        mask_buffers = make_mask_buffers(
            make_global_view,
            MASK_BUFFER_COUNT,
            MASK_BUFFER_SIZES,
            MaskBuffer0,
            MaskBuffer1,
            MaskBuffer2,
            MaskBuffer3,
        )
        sLD = make_shared_view(pld, COMPUTE_LD_ELEMS, 1)
        sLD4 = fx.logical_divide(sLD, fx.make_layout(4, 1))
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
            mask_buffers,
            gload_i32,
            batch,
            head,
        )
        q_stage_elems = PAIR_BM * DQK
        q_all_stages_elems = 2 * q_stage_elems
        probability_wave_elems = 64 * 16
        probability_stage_elems = PAIR_LOGICAL_WAVES * probability_wave_elems
        pdo = pqdo + fx.Int32(q_all_stages_elems)
        loader_tid = (wave - fx.Int32(PAIR_LOGICAL_WAVES)) * fx.Int32(64) + lane

        def stage_operand(
            view,
            destination,
            qchunk,
            stage,
            dim: fx.Constexpr[int],
            load_iterations: fx.Constexpr[int],
        ):
            qbase = qchunk * fx.Int32(PAIR_BM)
            stage_offset = fx.Int32(stage * PAIR_BM * dim)
            for load_step in fx.range_constexpr(load_iterations):
                linear = fx.Int32(load_step * PAIR_LOGICAL_WAVES * 64) + loader_tid
                lds_offset = linear * fx.Int32(VPT)
                subtile = lds_offset // fx.Int32(8 * 32)
                within_subtile = lds_offset % fx.Int32(8 * 32)
                row_in_group = within_subtile // fx.Int32(32)
                column_in_subtile = within_subtile % fx.Int32(32)
                row_group = subtile // fx.Int32(dim // 32)
                d_subtile = subtile % fx.Int32(dim // 32)
                row = row_group * fx.Int32(8) + row_in_group
                column = d_subtile * fx.Int32(32) + b_operand_column(row_group, column_in_subtile)
                source = fx.logical_divide(fx.slice(view, (qbase + row, None)), fx.make_layout(VPT, 1))
                fx.copy(
                    dma128,
                    fx.slice(source, (None, column // fx.Int32(VPT))),
                    dma_destination(destination, stage_offset + lds_offset),
                )

        def stage_metadata(qchunk, stage):
            qbase = qchunk * fx.Int32(PAIR_BM)
            stat_base = fx.Int32(stage * 2 * PAIR_BM)
            if const_expr(LSE_IN_LOG2):
                if loader_tid < fx.Int32(PAIR_BM // 4):
                    source_pack = qchunk * fx.Int32(PAIR_BM // 4) + loader_tid
                    destination_pack = fx.Int32(stage * (PAIR_BM // 2)) + loader_tid
                    fx.copy(dma128, fx.slice(gLSE4, (None, source_pack)), fx.slice(sLD4, (None, destination_pack)))
                    fx.copy(
                        dma128,
                        fx.slice(gDEL4, (None, source_pack)),
                        fx.slice(sLD4, (None, destination_pack + fx.Int32(PAIR_BM // 4))),
                    )
            elif loader_tid < fx.Int32(PAIR_BM):
                lse = gload_f32(gLSE, qbase + loader_tid)
                lse_log2 = lse * _f32(_LOG2E)
                lse_log2 = (lse < _f32(_NEG_BIG)).select(_f32(0.0), lse_log2)
                fx.ptr_store(lse_log2, pld + stat_base + loader_tid)
                delta = gload_f32(gDEL, qbase + loader_tid)
                fx.ptr_store(delta, pld + stat_base + fx.Int32(PAIR_BM) + loader_tid)

        def stage_q_metadata(qchunk, stage):
            stage_operand(gQ, pqdo, qchunk, stage, DQK, PAIR_Q_LOAD_IT)
            stage_metadata(qchunk, stage)

        def stage_do(qchunk, stage):
            stage_operand(gDO, pdo, qchunk, stage, DV, PAIR_DO_LOAD_IT)

        def stage_qdo_metadata(qchunk, stage):
            stage_q_metadata(qchunk, stage)
            stage_do(qchunk, stage)

        def load_score_a(base, stage, k_step, dim: fx.Constexpr[int]):
            stage_offset = fx.Int32(stage * PAIR_BM * dim)
            d_subtile = fx.Int32(k_step) // fx.Int32(2)
            d_half = fx.Int32(k_step) % fx.Int32(2)
            row_group = lane_row // fx.Int32(8)
            row_in_group = lane_row % fx.Int32(8)
            column = d_half * fx.Int32(16) + lane_half * fx.Int32(VPT)
            element_offset = (
                stage_offset
                + (row_group * fx.Int32(dim // 32) + d_subtile) * fx.Int32(8 * 32)
                + row_in_group * fx.Int32(32)
                + b_operand_column(row_group, column)
            )
            source = fx.make_view(fx.add_offset(base, fx.make_int_tuple(element_offset)), fx.make_layout(VPT, 1))
            return source.load()

        def load_update_a(
            base,
            stage,
            q_step,
            d_chunk,
            dim: fx.Constexpr[int],
        ):
            stage_offset = fx.Int32(stage * PAIR_BM * dim)
            row_offset = lane % fx.Int32(16) // fx.Int32(4) + lane_half * fx.Int32(4)
            column_offset = lane % fx.Int32(4) * fx.Int32(4) + lane_row // fx.Int32(16) * fx.Int32(16)
            lane_offset = row_offset * fx.Int32(32) + column_offset
            halves = []
            for half in fx.range_constexpr(2):
                subtile = (
                    (fx.Int32(q_step) * fx.Int32(2) + fx.Int32(half)) * fx.Int32(dim // 32)
                    + fx.Int32(d_chunk)
                )
                source = fx.make_view(
                    fx.add_offset(
                        base,
                        fx.make_int_tuple(
                            stage_offset
                            + subtile * fx.Int32(8 * 32)
                            + lane_offset
                            - column_offset
                            + b_operand_column(q_step * 2 + half, column_offset)
                        ),
                    ),
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_rmem_tensor(4, fx.BFloat16)
                fx.copy(tr16, source, fragment)
                halves.append(fx.Vector(fragment.load()))
            return halves[0].shuffle(halves[1], list(range(8))).ir_value()

        def probability_pointer(stage, half):
            element_offset = (
                fx.Int32(stage * probability_stage_elems)
                + logical_wave * fx.Int32(probability_wave_elems)
                + lane * fx.Int32(16)
                + fx.Int32(half * 8)
            )
            return pprob + element_offset

        def store_probability(stage, first, second):
            fx.ptr_store(first, probability_pointer(stage, 0))
            fx.ptr_store(second, probability_pointer(stage, 1))

        def load_probability(stage, half):
            return fx.ptr_load(probability_pointer(stage, half), result_type=fx.Vector.make_type(8, fx.BFloat16))

        def qchunk_at(
            tile_index,
            first_tile,
            indices,
            list_row,
            indexed: fx.Constexpr[bool],
        ):
            if const_expr(indexed):
                list_index = tile_index // fx.Int32(PAIR_LIST_Q_SPLIT)
                within_list_tile = tile_index % fx.Int32(PAIR_LIST_Q_SPLIT)
                return (
                    gload_i32(indices, list_row + list_index) * fx.Int32(PAIR_LIST_Q_SPLIT) + within_list_tile
                )
            return first_tile + tile_index

        def tile_is_masked(tile_index, mask_mode: fx.Constexpr[int]):
            if const_expr(mask_mode == _MASK_NONE):
                return False
            if const_expr(mask_mode == _MASK_ALL):
                return True
            return tile_index < fx.Int32(PAIR_CAUSAL_TILES)

        zero16 = fx.Vector.filled(16, 0.0, fx.Float32).ir_value()
        if producer_wave:
            k_packs = []
            v_packs = []
            for k_step in fx.range_constexpr(max(DQK, DV) // 16):
                column = fx.Int32(k_step * 16) + lane_half * fx.Int32(VPT)
                if const_expr(k_step < DQK // 16):
                    k_packs.append(load_global_pack(gK, key_pos, column))
                if const_expr(k_step < DV // 16):
                    v_packs.append(load_global_pack(gV, key_pos, column))
            dk_acc = [fx.make_rmem_tensor(16, fx.Float32) for _ in fx.range_constexpr(DQK // 32)]
            for d_chunk in fx.range_constexpr(DQK // 32):
                dk_acc[d_chunk].fill(0)

            def produce_tile(qchunk, masked, stage):
                score = zero16
                dp = zero16
                for k_step in fx.range_constexpr(DQK // 16):
                    score = mfma(load_score_a(pqdo, stage, k_step, DQK), k_packs[k_step], score)
                for k_step in fx.range_constexpr(DV // 16):
                    dp = mfma(load_score_a(pdo, stage, k_step, DV), v_packs[k_step], dp)

                q_lane_base = fx.Int32(4) * lane_half
                stat_base = fx.Int32(stage * 2 * PAIR_BM)
                prefetch_all_metadata = PAIR_MC <= 32
                prefetch_lse = PAIR_MC >= 64
                first_lse_values = []
                first_delta_values = []
                if const_expr(prefetch_all_metadata or prefetch_lse):
                    for element in fx.range_constexpr(4):
                        qlocal = q_lane_base + fx.Int32(DQ32_CE[element])
                        first_lse_values.append(fx.Float32(fx.ptr_load(pld + stat_base + qlocal)))
                        if const_expr(prefetch_all_metadata):
                            first_delta_values.append(
                                fx.Float32(fx.ptr_load(pld + stat_base + fx.Int32(PAIR_BM) + qlocal))
                            )
                schedule_score_pipeline(
                    mfma_count=(DQK + DV) // 16,
                    dsrd_count=(DQK + DV) // 16 + (2 if prefetch_all_metadata else 1 if prefetch_lse else 0),
                    vmem_count=0,
                )

                score_values = fx.Vector(score)
                dp_values = fx.Vector(dp)
                first_probabilities = []
                first_ds_values = []
                for element in fx.range_constexpr(8):
                    qlocal = q_lane_base + fx.Int32(DQ32_CE[element])
                    qpos = qchunk * fx.Int32(PAIR_BM) + qlocal
                    if const_expr((prefetch_all_metadata or prefetch_lse) and element < 4):
                        lse_log2 = first_lse_values[element]
                    else:
                        lse_log2 = fx.Float32(fx.ptr_load(pld + stat_base + qlocal))
                    if const_expr(prefetch_all_metadata and element < 4):
                        delta = first_delta_values[element]
                    else:
                        delta = fx.Float32(fx.ptr_load(pld + stat_base + fx.Int32(PAIR_BM) + qlocal))
                    probability = _exp2(_f32(score_values[element]) * _f32(SC2) - lse_log2)
                    if const_expr(WINDOW_MASK):
                        keep = (qpos >= key_pos) & (qpos - key_pos < fx.Int32(WINDOW_SIZE))
                        probability = keep.select(probability, _f32(0.0))
                    elif const_expr(CAUSAL_MASK) and masked:
                        probability = (qpos >= key_pos).select(probability, _f32(0.0))
                    elif masked:
                        probability = evaluate_mask(qpos, key_pos).select(probability, _f32(0.0))
                    first_probabilities.append(probability)
                    first_ds_values.append(probability * (_f32(dp_values[element]) - delta))
                first_probability = (
                    fx.Vector.from_elements(first_probabilities, fx.Float32).to(fx.BFloat16).ir_value()
                )
                first_ds_fragment = make_fragment(
                    fx.Vector.from_elements(first_ds_values, fx.Float32).to(fx.BFloat16).ir_value(),
                    8,
                    fx.BFloat16,
                )
                first_update_fragments = []
                for d_chunk in fx.range_constexpr(DQK // 32):
                    first_update_fragments.append(
                        make_fragment(load_update_a(pqdo, stage, 0, d_chunk, DQK), 8, fx.BFloat16)
                    )

                prefetch_second_metadata = PAIR_MC >= 64
                second_lse_values = []
                second_delta_values = []
                if const_expr(prefetch_second_metadata):
                    for element in fx.range_constexpr(4):
                        score_index = 8 + element
                        qlocal = q_lane_base + fx.Int32(DQ32_CE[score_index])
                        second_lse_values.append(fx.Float32(fx.ptr_load(pld + stat_base + qlocal)))
                        second_delta_values.append(
                            fx.Float32(fx.ptr_load(pld + stat_base + fx.Int32(PAIR_BM) + qlocal))
                        )
                schedule_pack0_pipeline(
                    vmem_count=0,
                    exp_count=8,
                    dsrd_count=4 * (DQK // 32) + (2 if prefetch_second_metadata else 0),
                )

                second_probabilities = []
                second_ds_values = []
                for element in fx.range_constexpr(8):
                    score_index = 8 + element
                    qlocal = q_lane_base + fx.Int32(DQ32_CE[score_index])
                    qpos = qchunk * fx.Int32(PAIR_BM) + qlocal
                    if const_expr(prefetch_second_metadata and element < 4):
                        lse_log2 = second_lse_values[element]
                        delta = second_delta_values[element]
                    else:
                        lse_log2 = fx.Float32(fx.ptr_load(pld + stat_base + qlocal))
                        delta = fx.Float32(fx.ptr_load(pld + stat_base + fx.Int32(PAIR_BM) + qlocal))
                    probability = _exp2(_f32(score_values[score_index]) * _f32(SC2) - lse_log2)
                    if const_expr(WINDOW_MASK):
                        keep = (qpos >= key_pos) & (qpos - key_pos < fx.Int32(WINDOW_SIZE))
                        probability = keep.select(probability, _f32(0.0))
                    elif const_expr(CAUSAL_MASK) and masked:
                        probability = (qpos >= key_pos).select(probability, _f32(0.0))
                    elif masked:
                        probability = evaluate_mask(qpos, key_pos).select(probability, _f32(0.0))
                    second_probabilities.append(probability)
                    second_ds_values.append(probability * (_f32(dp_values[score_index]) - delta))
                second_probability = (
                    fx.Vector.from_elements(second_probabilities, fx.Float32).to(fx.BFloat16).ir_value()
                )
                second_ds_fragment = make_fragment(
                    fx.Vector.from_elements(second_ds_values, fx.Float32).to(fx.BFloat16).ir_value(),
                    8,
                    fx.BFloat16,
                )
                store_probability(stage, first_probability, second_probability)
                fx.rocdl.sched_dswr(2)

                second_update_fragments = []
                for d_chunk in fx.range_constexpr(min(2, DQK // 32)):
                    second_update_fragments.append(
                        make_fragment(load_update_a(pqdo, stage, 1, d_chunk, DQK), 8, fx.BFloat16)
                    )
                for d_chunk in fx.range_constexpr(DQK // 32):
                    fx.gemm(atom, dk_acc[d_chunk], first_update_fragments[d_chunk], first_ds_fragment, dk_acc[d_chunk])
                schedule_pack1_pipeline(
                    mfma_count=DQK // 32,
                    exp_count=8,
                    dsrd_count=2 * (DQK // 32),
                )
                for d_chunk in fx.range_constexpr(DQK // 32):
                    fx.gemm(
                        atom,
                        dk_acc[d_chunk],
                        second_update_fragments[d_chunk],
                        second_ds_fragment,
                        dk_acc[d_chunk],
                    )
                    if const_expr(d_chunk + 2 < DQK // 32):
                        second_update_fragments.append(
                            make_fragment(load_update_a(pqdo, stage, 1, d_chunk + 2, DQK), 8, fx.BFloat16)
                        )
                schedule_update_tail(mfma_count=DQK // 32, dsrd_count=2 * (DQK // 32))
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()

            def run_producer(
                tile_count,
                first_tile,
                indices,
                list_row,
                indexed: fx.Constexpr[bool],
                mask_mode: fx.Constexpr[int],
            ):
                if tile_count > fx.Int32(0):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    for tile_index in range(fx.Int32(0), tile_count, fx.Int32(1)):
                        qchunk = qchunk_at(tile_index, first_tile, indices, list_row, indexed)
                        produce_tile(qchunk, tile_is_masked(tile_index, mask_mode), tile_index % fx.Int32(2))

            direct_mask = DENSE_MASK or CAUSAL_MASK or WINDOW_MASK
            if const_expr(direct_mask):
                if const_expr(DENSE_MASK):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    for qpair in range(fx.Int32(0), fx.Int32(PAIR_MC), fx.Int32(2)):
                        produce_tile(qpair, False, 0)
                        produce_tile(qpair + fx.Int32(1), False, 1)
                elif const_expr(WINDOW_MASK):
                    first_tile = nc * fx.Int32(PAIR_CAUSAL_TILES)
                    remaining = fx.Int32(PAIR_MC) - first_tile
                    tile_count = (remaining < fx.Int32(PAIR_WINDOW_TILES)).select(
                        remaining, fx.Int32(PAIR_WINDOW_TILES)
                    )
                    run_producer(tile_count, first_tile, gIP, fx.Int32(0), False, _MASK_ALL)
                else:
                    first_tile = nc * fx.Int32(PAIR_CAUSAL_TILES)
                    tile_count = fx.Int32(PAIR_MC) - first_tile
                    run_producer(
                        tile_count,
                        first_tile,
                        gIP,
                        fx.Int32(0),
                        False,
                        _MASK_CAUSAL_BOUNDARY,
                    )
            else:
                list_nc = nc // fx.Int32(PAIR_KEY_SPLIT)
                list_row = list_nc * fx.Int32(MC)
                partial_count = gload_i32(gCP, list_nc) * fx.Int32(PAIR_LIST_Q_SPLIT)
                full_count = gload_i32(gCF, list_nc) * fx.Int32(PAIR_LIST_Q_SPLIT)
                run_producer(partial_count, fx.Int32(0), gIP, list_row, True, _MASK_ALL)
                run_producer(full_count, fx.Int32(0), gIF, list_row, True, _MASK_NONE)

            output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
            output_column_base = fx.Int32(4) * lane_half
            output_row = fx.logical_divide(fx.slice(gDK, (key_pos, None)), fx.make_layout(4, 1))
            scale_vector = fx.Vector.filled(16, SCALE, fx.Float32)
            for d_chunk in fx.range_constexpr(DQK // 32):
                values_bf16 = (fx.Vector(dk_acc[d_chunk].load()) * scale_vector).to(fx.BFloat16)
                for pack in fx.range_constexpr(4):
                    output = fx.Vector.from_elements(
                        [values_bf16[pack * 4 + element] for element in fx.range_constexpr(4)],
                        fx.BFloat16,
                    )
                    output_fragment.store(output.ir_value())
                    column = fx.Int32(d_chunk * 32) + output_column_base + fx.Int32(pack * 8)
                    fx.copy(o64, output_fragment, fx.slice(output_row, (None, column // fx.Int32(4))))
        else:
            dv_acc = [fx.make_rmem_tensor(16, fx.Float32) for _ in fx.range_constexpr(DV // 32)]
            for d_chunk in fx.range_constexpr(DV // 32):
                dv_acc[d_chunk].fill(0)

            def prefetch_dv_pack(stage, half: fx.Constexpr[int]):
                probability = make_fragment(load_probability(stage, half), 8, fx.BFloat16)
                update_fragments = []
                for d_chunk in fx.range_constexpr(DV // 32):
                    update_fragments.append(
                        make_fragment(load_update_a(pdo, stage, half, d_chunk, DV), 8, fx.BFloat16)
                    )
                if const_expr(half == 0):
                    fx.rocdl.sched_dsrd(1 + 2 * (DV // 32))
                    schedule_fence()
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                return probability, update_fragments

            def consume_dv_tile(
                first_probability,
                second_probability,
                first_update_fragments,
                second_update_fragments,
                overlap_q_metadata: fx.Constexpr[bool],
            ):
                for d_chunk in fx.range_constexpr(DV // 32):
                    fx.gemm(
                        atom,
                        dv_acc[d_chunk],
                        first_update_fragments[d_chunk],
                        first_probability,
                        dv_acc[d_chunk],
                    )
                second_read_count = 1 + 2 * (DV // 32)
                preload_count = min(3, second_read_count)
                remaining_reads = second_read_count - preload_count
                fx.rocdl.sched_dsrd(preload_count)
                for mfma_index in fx.range_constexpr(DV // 32):
                    fx.rocdl.sched_mfma(1)
                    reads = min(2, max(0, remaining_reads - 2 * mfma_index))
                    if const_expr(reads):
                        fx.rocdl.sched_dsrd(reads)
                schedule_fence()
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                for d_chunk in fx.range_constexpr(DV // 32):
                    fx.gemm(
                        atom,
                        dv_acc[d_chunk],
                        second_update_fragments[d_chunk],
                        second_probability,
                        dv_acc[d_chunk],
                    )
                next_vmem_count = PAIR_Q_LOAD_IT + 2 if overlap_q_metadata else 0
                for mfma_index in fx.range_constexpr(DV // 32):
                    if const_expr(mfma_index < next_vmem_count):
                        fx.rocdl.sched_vmem(1)
                    fx.rocdl.sched_mfma(1)
                for _ in fx.range_constexpr(max(0, next_vmem_count - DV // 32)):
                    fx.rocdl.sched_vmem(1)
                schedule_fence()

            def run_consumer(
                tile_count,
                first_tile,
                indices,
                list_row,
                indexed: fx.Constexpr[bool],
            ):
                if tile_count > fx.Int32(0):
                    stage_qdo_metadata(qchunk_at(fx.Int32(0), first_tile, indices, list_row, indexed), 0)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    fx.gpu.barrier()
                    for tile_index in range(fx.Int32(0), tile_count, fx.Int32(1)):
                        if tile_index == fx.Int32(0):
                            if tile_count > fx.Int32(1):
                                stage_qdo_metadata(
                                    qchunk_at(fx.Int32(1), first_tile, indices, list_row, indexed),
                                    1,
                                )
                        else:
                            previous_stage = (tile_index - fx.Int32(1)) % fx.Int32(2)
                            first_probability, first_updates = prefetch_dv_pack(previous_stage, 0)
                            second_probability, second_updates = prefetch_dv_pack(previous_stage, 1)
                            has_next = tile_index + fx.Int32(1) < tile_count
                            if has_next:
                                next_qchunk = qchunk_at(
                                    tile_index + fx.Int32(1),
                                    first_tile,
                                    indices,
                                    list_row,
                                    indexed,
                                )
                                stage_q_metadata(next_qchunk, previous_stage)
                                consume_dv_tile(
                                    first_probability,
                                    second_probability,
                                    first_updates,
                                    second_updates,
                                    True,
                                )
                                stage_do(next_qchunk, previous_stage)
                                fx.rocdl.sched_vmem(PAIR_DO_LOAD_IT)
                                schedule_fence()
                            else:
                                consume_dv_tile(
                                    first_probability,
                                    second_probability,
                                    first_updates,
                                    second_updates,
                                    False,
                                )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.gpu.barrier()
                    last_stage = (tile_count - fx.Int32(1)) % fx.Int32(2)
                    first_probability, first_updates = prefetch_dv_pack(last_stage, 0)
                    second_probability, second_updates = prefetch_dv_pack(last_stage, 1)
                    consume_dv_tile(
                        first_probability,
                        second_probability,
                        first_updates,
                        second_updates,
                        False,
                    )

            direct_mask = DENSE_MASK or CAUSAL_MASK or WINDOW_MASK
            if const_expr(direct_mask):
                if const_expr(DENSE_MASK):
                    first_tile = fx.Int32(0)
                    tile_count = fx.Int32(PAIR_MC)
                elif const_expr(WINDOW_MASK):
                    first_tile = nc * fx.Int32(PAIR_CAUSAL_TILES)
                    remaining = fx.Int32(PAIR_MC) - first_tile
                    tile_count = (remaining < fx.Int32(PAIR_WINDOW_TILES)).select(
                        remaining, fx.Int32(PAIR_WINDOW_TILES)
                    )
                else:
                    first_tile = nc * fx.Int32(PAIR_CAUSAL_TILES)
                    tile_count = fx.Int32(PAIR_MC) - first_tile
                run_consumer(tile_count, first_tile, gIP, fx.Int32(0), False)
            else:
                list_nc = nc // fx.Int32(PAIR_KEY_SPLIT)
                list_row = list_nc * fx.Int32(MC)
                partial_count = gload_i32(gCP, list_nc) * fx.Int32(PAIR_LIST_Q_SPLIT)
                full_count = gload_i32(gCF, list_nc) * fx.Int32(PAIR_LIST_Q_SPLIT)
                run_consumer(partial_count, fx.Int32(0), gIP, list_row, True)
                run_consumer(full_count, fx.Int32(0), gIF, list_row, True)

            output_fragment = fx.make_rmem_tensor(4, fx.BFloat16)
            output_column_base = fx.Int32(4) * lane_half
            output_row = fx.logical_divide(fx.slice(gDV, (key_pos, None)), fx.make_layout(4, 1))
            for d_chunk in fx.range_constexpr(DV // 32):
                values_bf16 = fx.Vector(dv_acc[d_chunk].load()).to(fx.BFloat16)
                for pack in fx.range_constexpr(4):
                    output = fx.Vector.from_elements(
                        [values_bf16[pack * 4 + element] for element in fx.range_constexpr(4)],
                        fx.BFloat16,
                    )
                    output_fragment.store(output.ir_value())
                    column = fx.Int32(d_chunk * 32) + output_column_base + fx.Int32(pack * 8)
                    fx.copy(o64, output_fragment, fx.slice(output_row, (None, column // fx.Int32(4))))

    return _emit_dkdv_mfma32_body
