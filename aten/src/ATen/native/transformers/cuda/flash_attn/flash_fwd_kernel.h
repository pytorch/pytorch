/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include <ATen/native/transformers/cuda/flash_attn/block_info.h>
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>
#include <ATen/native/transformers/cuda/flash_attn/softmax.h>
#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>

namespace pytorch_flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    if (Is_first) {
        pytorch_flash::template reduce_max</*zero_init=*/true>(scores, scores_max);
        pytorch_flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        pytorch_flash::reduce_sum(scores, scores_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        pytorch_flash::template reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) {
            float scores_max_cur = !Check_inf
                ? scores_max(mi)
                : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            scores_sum(mi) *= scores_scale;
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
        }
        pytorch_flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        pytorch_flash::reduce_sum(scores, scores_sum_cur);
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename TiledCopy>
inline __device__ void write_softmax_to_gmem(
    Tensor<Engine0, Layout0> const &tOrP, Tensor<Engine1, Layout1> &tPgP, TiledCopy gmem_tiled_copy_P
) {
    // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
    Layout l = tOrP.layout();
    Tensor tPrP = make_tensor(tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
    CUTE_STATIC_ASSERT_V(size<2>(tPgP) == _1{});
    CUTE_STATIC_ASSERT_V(size<1>(tPrP) == size<1>(tPgP));
    #pragma unroll
    for (int mi = 0; mi < size<1>(tPrP); ++mi) {
        cute::copy(gmem_tiled_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_M = kBlockM / decltype(size<0>(typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q || binfo.actual_seqlen_k == 0) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
        // We exit early and write 0 to gO and gLSE.
        // Otherwise we might read OOB elements from gK and gV.
        if (n_block_max <= n_block_min) {
            // Save seed and offset for backward. If we don't have this here, the 0-th thread block might
            // exit early and no one saves the rng state.
            if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
                auto seeds = at::cuda::philox::unpack(params.philox_args);
                if (params.philox_args.captured_) {
                    *params.seed = std::get<0>(seeds);
                    *params.extragraph_offset = std::get<1>(seeds);
                }
            }
            const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
                + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
            const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
            Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                                    Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_stride(params.o_row_stride, _1{}));
            Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

            typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
            auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
            Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
            Tensor tOrO = make_tensor<Element>(shape(tOgO));
            clear(tOrO);
            // Construct identity layout for sO
            Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            // Repeat the partitioning with identity layouts
            Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
            if (!Is_even_K) {
                #pragma unroll
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
            }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
            );
            #pragma unroll
            for (int m = 0; m < size<1>(tOgO); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
            }
            return;
        }
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyP gmem_tiled_copy_P;
    auto gmem_thr_copy_P = gmem_tiled_copy_P.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tPgP = gmem_thr_copy_P.partition_D(gP);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // TODO: this might need to change if we change the mma instruction in SM70
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    Tensor tQrQ = make_fragment_like(tQgQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // Copy rmem to smem
    // // copy(tQrQ, tQsQ);
    // pytorch_flash::cp_async_wait<0>();
    // __syncthreads();
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        pytorch_flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    auto seeds = at::cuda::philox::unpack(params.philox_args);
    if (params.philox_args.captured_) {
        *params.seed = std::get<0>(seeds);
        *params.extragraph_offset = std::get<1>(seeds);
    }

    unsigned long long seed = std::get<0>(seeds);
    unsigned long long offset = std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32;

    clear(acc_o);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();

        pytorch_flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // if (cute::thread0()) { print_tensor(scores); }
        // We don't put the masking before the matmul S = Q K^T because we don't clear sK
        // for rows outside actual_seqlen_k. So those rows could have Inf / NaN, and the matmul
        // can produce Inf / NaN.
        if (!Is_causal && !Is_local) {
            if (!Is_even_MN) { pytorch_flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN); }
        } else {
            // Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
            // Tensor taccScS = thr_mma.partition_C(caccS);                           // (MMA,MMA_M,MMA_N)
            // static_assert(decltype(size<0>(taccScS))::value == 4);
            // // Convert to ((2, 2), MMA_M, MMA_N) then take only the row indices.
            // Tensor idx_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
            // Tensor idx_rowcol = make_tensor(taccScS.data(), pytorch_flash::convert_layout_acc_rowcol(taccScS.layout()));
            // pytorch_flash::apply_mask_causal_w_idx(scores, idx_rowcol, n_block * kBlockN, binfo.actual_seqlen_k,
            //                               m_block * kBlockM);
            // Idk why it's get<1> and not get<0> of the stride.
            // if (cute::thread0()) { print(idx_row.layout()); print(stride<1>(idx_row)); printf("stride = %d \n", get<1>(stride<1>(idx_row))); }
            // I can't get the stride from idx_row
            pytorch_flash::apply_mask_local</*HasWSLeft=*/Is_local>(
                scores, n_block * kBlockN, binfo.actual_seqlen_k,
                // m_block * kBlockM + get<0>(idx_row(0)),
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                binfo.actual_seqlen_q, kNWarps * 16,
                params.window_size_left, params.window_size_right
                // m_block * kBlockM + (tidx / 32) * 16, kNWarps * 16
                // m_block * kBlockM + (tidx / 32) * (kBlockM / kNWarps), 16
            );
            // if (cute::thread0()) { print_tensor(scores); }
        }

        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2)
            : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        // Convert scores from fp32 to fp16/bf16
        Tensor rP = pytorch_flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor tOrP_copy = make_fragment_like(tOrP);
            cute::copy(tOrP, tOrP_copy);
            pytorch_flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                tOrP_copy, params.p_dropout_in_uint8_t, seed, offset,
                block_row_idx, block_col_idx, kNWarps
            );
            pytorch_flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
            tPgP.data() = tPgP.data() + (-kBlockN);
        }
        if (Is_dropout) {
            pytorch_flash::apply_dropout(tOrP, params.p_dropout_in_uint8_t, seed, offset,
                                 block_row_idx, block_col_idx, kNWarps);
        }
        // if (cute::thread0()) { print(tOrP); }

        pytorch_flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        pytorch_flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Is_local && n_block * kBlockN < (m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right) {
            pytorch_flash::apply_mask_local(
                scores, n_block * kBlockN, binfo.actual_seqlen_k,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                binfo.actual_seqlen_q, kNWarps * 16,
                params.window_size_left, params.window_size_right
            );
        }
        softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        Tensor rP = pytorch_flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor tOrP_copy = make_fragment_like(tOrP);
            cute::copy(tOrP, tOrP_copy);
            pytorch_flash::apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                tOrP_copy, params.p_dropout_in_uint8_t, seed, offset,
                block_row_idx, block_col_idx, kNWarps
            );
            pytorch_flash::write_softmax_to_gmem(tOrP_copy, tPgP, gmem_tiled_copy_P);
            tPgP.data() = tPgP.data() + (-kBlockN);
        }
        if (Is_dropout) {
            pytorch_flash::apply_dropout(tOrP, params.p_dropout_in_uint8_t, seed, offset,
                                 block_row_idx, block_col_idx, kNWarps);
        }

        pytorch_flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
    Tensor lse = make_fragment_like(scores_sum);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? INFINITY : scores_max(mi) * params.scale_softmax + __logf(sum);
        float scale = !Is_dropout ? inv_sum : inv_sum * params.rp_dropout;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }

    // if (cute::thread0()) { print(acc_o_rowcol); }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = pytorch_flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyOaccum,
        typename Kernel_traits::GmemTiledCopyO
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                     make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
        + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
        + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // TODO: this might need to change if we change the mma instruction in SM70
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.
    typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
    auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
    auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
    if constexpr (Append_KV) {
        // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
        // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
        // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
        const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN) * (params.rotary_dim / 2);
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
                                  make_stride(params.rotary_dim / 2, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                      Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                      make_stride(params.rotary_dim / 2, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
        // if (cute::thread(8, 0)) { print_tensor(gCos); }
        // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

        const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
            + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
        const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
            + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
        // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
        // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
        // This maps to accessing the first 64 rows of knew_ptr.
        Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
                                                + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.knew_row_stride, _1{}));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
        Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
                                                + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
                                  Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                  make_stride(params.vnew_row_stride, _1{}));
        Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
        Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

        const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
        for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
            pytorch_flash::copy_w_min_idx<Is_even_K>(
                tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
            );
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
            if (params.rotary_dim == 0) {
                pytorch_flash::copy_w_min_idx<Is_even_K>(
                    tKgKnew, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
                );
            } else {
                if (params.is_rotary_interleaved) {
                    // Don't clear OOB_K because we're writing to global memory
                    pytorch_flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
                } else {
                    // Don't clear OOB_K because we're writing to global memory
                    pytorch_flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
                        tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
                        binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
                    );
                    tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
                    tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));

                }
            }
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        if (n_block_max > n_block_copy_min) {
            tKgK.data() = tKgK.data() + (n_block_max - n_block_copy_min) * kBlockN * params.k_row_stride;
            tVgV.data() = tVgV.data() + (n_block_max - n_block_copy_min) * kBlockN * params.v_row_stride;
        }
    }

    // Read Q from gmem to smem, optionally apply rotary embedding.
    Tensor tQrQ = make_fragment_like(tQgQ);
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    } else {
        const index_t row_offset_cossin = (binfo.seqlen_k_cache + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
        // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
        // We do this by setting the row stride of gCos / gSin to 0.
        Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
        Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
        Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
        Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
        Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
        if (params.is_rotary_interleaved) {
            pytorch_flash::copy_rotary_interleaved<Is_even_K>(
                tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        } else {
            pytorch_flash::copy_rotary_contiguous<Is_even_K>(
                tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
                0, params.d, params.rotary_dim
            );
        }
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    pytorch_flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();

    // pytorch_flash::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();

    clear(acc_o);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        cute::cp_async_fence();

        pytorch_flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        // if (cute::thread0()) { print(scores); }
        // We don't put the masking before the matmul S = Q K^T because we don't clear sK
        // for rows outside actual_seqlen_k. So those rows could have Inf / NaN, and the matmul
        // can produce Inf / NaN.
        if (!Is_causal && !Is_local) {
            if (!Is_even_MN) { pytorch_flash::apply_mask(scores, binfo.actual_seqlen_k - n_block * kBlockN); }
        } else {
            pytorch_flash::apply_mask_local(scores, n_block * kBlockN, binfo.actual_seqlen_k,
                                    m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                                    binfo.actual_seqlen_q, kNWarps * 16,
                                    params.window_size_left, params.window_size_right
                                    );
        }

        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
        // __syncthreads();

        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2)
            : softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert scores from fp32 to fp16/bf16
        Tensor rP = pytorch_flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        pytorch_flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        // Advance gV
        tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
        pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        cute::cp_async_fence();

        pytorch_flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        pytorch_flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
            pytorch_flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), pytorch_flash::convert_layout_acc_rowcol(acc_s.layout()));
        if (Is_local && n_block * kBlockN < (m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right) {
            pytorch_flash::apply_mask_local(
                scores, n_block * kBlockN, binfo.actual_seqlen_k,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                binfo.actual_seqlen_q, kNWarps * 16,
                params.window_size_left, params.window_size_right
            );
        }
        softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(scores, scores_max, scores_sum, acc_o, params.scale_softmax_log2);

        Tensor rP = pytorch_flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), pytorch_flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        pytorch_flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), pytorch_flash::convert_layout_acc_rowcol(acc_o.layout()));
    // if (cute::thread0()) { print(acc_o_rowcol); }
    Tensor lse = make_fragment_like(scores_sum);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : scores_max(mi) * params.scale_softmax + __logf(sum);
        float scale = inv_sum;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }
    // if (cute::thread0()) { print(lse); }
    // if (cute::thread0()) { print(acc_o_rowcol); }

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = pytorch_flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split) { __syncthreads(); }

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    pytorch_flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    // __syncthreads();
    // if (cute::thread0()) { print(tOgOaccum); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    pytorch_flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Is_even_MN, bool Is_even_K, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    pytorch_flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Is_even_MN, Is_even_K, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(params.b * params.h * params.seqlen_q, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then tranpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < params.b * params.h * params.seqlen_q - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
    }
    // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) { gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum; }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        pytorch_flash::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
        }
        tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
    }
    // if (cute::thread0()) { print_tensor(tOrO); }

    Tensor rO = pytorch_flash::convert_type<Element>(tOrO);
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < params.b * params.h * params.seqlen_q) {
            const int batch_idx = idx / (params.h * params.seqlen_q);
            const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
            // The index to the rows of Q
            const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    // TODO: Should check if this is using vectorized store, but it seems pretty fast
                    copy(rO(_, m, k), gO);
                    // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
                    // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
