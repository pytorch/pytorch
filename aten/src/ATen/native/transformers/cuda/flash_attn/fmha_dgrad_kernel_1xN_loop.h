/* Copyright (c) 2022, Tri Dao.
 */

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/transformers/cuda/flash_attn/fmha_fprop_kernel_1xN.h>
#include <ATen/native/transformers/cuda/flash_attn/fmha_kernel.h>
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>
#include <ATen/native/transformers/cuda/flash_attn/gemm.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int ROWS, int THREADS_PER_ROW, typename elem_type=__half, int M, typename Gmem_softmax_sum>
inline __device__ void dot_do_o(const uint4 (&do_)[M], const uint4 (&o)[M], const float scale,
                                Gmem_softmax_sum gmem_softmax_d, int tidx) {
    float sum[M];
    fmha::SumOp<float> sum_op;
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        sum[mi] = fmha::Allreduce<THREADS_PER_ROW>::run(
            fmha::hmulsum8<elem_type>(do_[mi], o[mi]), sum_op
        ) * scale;
    }
    const int dp_sum_row = tidx / THREADS_PER_ROW;
    if ((dp_sum_row < ROWS) && (tidx % THREADS_PER_ROW == 0)) {
        gmem_softmax_d.store_row(reinterpret_cast<const uint32_t (&)[M]>(sum), dp_sum_row);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Just compute dot(do, o) and write the result (softmax_d) to global memory as a separate kernel.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using elem_type = typename Kernel_traits::elem_type;
#else
    constexpr bool is_fp16_type = std::is_same<typename Kernel_traits::elem_type, __half>::value;
    assert(is_fp16_type);
    using elem_type = __half;
#endif

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 3rd batched GEMM.
    using Cta_tile_dkv =
        fmha::Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;

    static_assert(Cta_tile_dkv::N == 16 || Cta_tile_dkv::N == 32 || Cta_tile_dkv::N == 64 || Cta_tile_dkv::N == 128);
    static_assert(Cta_tile_dkv::K == 16);

    // The global memory tile to load dO.
    using Gmem_tile_do = typename Kernel_traits::Gmem_tile_do;

    // The global memory tile to load O.Loading O here is similar to loading dO.
    using Gmem_tile_o = Gmem_tile_do;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    // How many steps to jump per iteration.
    const int step_stride = gridDim.z;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    if( binfo.stop_early() ) return;

    // Allocate the global memory tile loader for dO.
    Gmem_tile_do gmem_do(params.do_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                         params.d, binfo, tidx, true);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                       params.d, binfo, tidx, true);

    Gmem_softmax_sum gmem_softmax_d(params.dsoftmax_sum, params, tidx);

    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    const int steps = (params.seqlen_q + Cta_tile_p::M - 1) / Cta_tile_p::M;
    // Wind gmem tiles to the correct position.
    gmem_do.move(blockIdx.z);
    gmem_o.move(blockIdx.z);
    gmem_softmax_d.move(blockIdx.z);

    // Load over the entire sequence length.
    for (int l = blockIdx.z; l < steps; l += step_stride) {
        if (l * Cta_tile_p::M  >= binfo.actual_seqlen_q)
            break;

        gmem_do.load();
        gmem_do.move(step_stride);
        gmem_o.load();
        gmem_o.move(step_stride);

        dot_do_o<Gmem_tile_do::ROWS, Gmem_tile_do::THREADS_PER_ROW, elem_type>(
            gmem_do.fetch_, gmem_o.fetch_, params.p_dropout, gmem_softmax_d, tidx
        );
        gmem_softmax_d.move(step_stride);
    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_first, bool Is_last, bool Seq_parallel=false, typename Params, typename Prng>
inline __device__ void compute_dq_dk_dv_1xN_one_iter(const Params &params, Prng &ph,
                                                     const int loop_step_idx) {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using elem_type = typename Kernel_traits::elem_type;
#else
    constexpr bool is_fp16_type = std::is_same<typename Kernel_traits::elem_type, __half>::value;
    assert(is_fp16_type);
    using elem_type = __half;
#endif

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_dq = typename Kernel_traits::Cta_tile_o;
    // The description of the CTA tile for the 3rd batched GEMM.
    using Cta_tile_dkv =
        fmha::Cta_tile_extd<Cta_tile_p::N, Cta_tile_p::K, Cta_tile_p::M, Cta_tile_p::WARPS_N, 1, Cta_tile_p::WARPS_M>;

    static_assert(Cta_tile_dkv::M == 512 ||  Cta_tile_dkv::M == 256 || Cta_tile_dkv::M == 128);
    static_assert(Cta_tile_dkv::N == 16 || Cta_tile_dkv::N == 32 || Cta_tile_dkv::N == 64 || Cta_tile_dkv::N == 128);
    static_assert(Cta_tile_dkv::K == 16);

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_dq = fmha::Hmma_tile<Cta_tile_dq>;
    // The MMA tile for the 3rd GEMM.
    using Mma_tile_dkv = fmha::Hmma_tile<Cta_tile_dkv>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    // The shared memory tile to reload Q transposed.
    using Smem_tile_qt = fmha::Smem_tile_b<Cta_tile_dkv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle K^T. Treat K^T as V
    using Smem_tile_kt = typename Kernel_traits::Smem_tile_v;

    // Treating V as K. We need to use Kernel_traits::Smem_tile_k otherwise loading will be wrong
    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_k;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_k;

    // The global memory tile to load dO.
    using Gmem_tile_do = typename Kernel_traits::Gmem_tile_do;
    // The shared memory tile to load dO.
    // Treating dO as Q.
    using Smem_tile_do = typename Kernel_traits::Smem_tile_q;
    // The shared memory tile to reload dO transposed.
    using Smem_tile_dot = fmha::Smem_tile_b<Cta_tile_dkv, fmha::Row, Gmem_tile_q::BYTES_PER_LDG, 2>;

    // The global memory tile to load O.Loading O here is similar to loading dO.
    using Gmem_tile_o = Gmem_tile_do;

    // The global memory tile to store dQ.
    using Gmem_tile_dq = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_dq_tmp = fmha::Gmem_tile_o<Cta_tile_dq, 4>;
    // The shared memory tile to swizzle dQ.
    using Smem_tile_dq = typename Kernel_traits::Smem_tile_o;

    // The global memory tile to store dV.
    using Gmem_tile_dv = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dV.
    using Smem_tile_dv = fmha::Smem_tile_mma_epilogue<Cta_tile_dkv>;

    // The global memory tile to store dK.
    using Gmem_tile_dk = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle dK.
    using Smem_tile_dk = fmha::Smem_tile_mma_epilogue<Cta_tile_dkv>;
    static_assert(Smem_tile_dk::NUM_LDS == Gmem_tile_dk::LDGS);
    static_assert(Smem_tile_dk::THREADS_PER_ROW == Gmem_tile_dk::THREADS_PER_ROW);

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Smem_tile_st = typename Kernel_traits::Smem_tile_st;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    // using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;
    using Gemm1 = Gemm_Q_K<Kernel_traits, /*K-in_regs=*/false, elem_type>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];
    // Shared memory layout if we keep V in registers:
    //  dO | Q | K / V | dQ | S | dP | dP_sum
    //  dV | dK
    // Shared memory layout if we keep V shared memory:
    //  dO | Q | K | V | dQ | S | dP | dP_sum
    //  dV | dK


    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    // if( binfo.stop_early() ) return;
    if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return;

    Gemm1 gemm_q_k(&smem_[Smem_tile_do::BYTES_PER_TILE], tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params.q_ptr, params.q_row_stride_in_elts, params.q_head_stride_in_elts,
                       params.d, binfo, tidx, true);
    // Allocate the global memory tile loader for dQ.
    Gmem_tile_dq gmem_dq(params.dq_ptr, params.dq_row_stride_in_elts, params.dq_head_stride_in_elts,
                         params.d, binfo, tidx);
    Gmem_tile_dq_tmp gmem_dq_tmp(params.o_tmp_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                                 params.d, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);

    fmha::Mask<Cta_tile_p, Is_causal> mask(binfo, tidx, loop_step_idx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params.k_ptr, params.k_row_stride_in_elts, params.k_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params.v_ptr, params.v_row_stride_in_elts, params.v_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_V];

    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);
    // Allocate the shared memory tile loader for K^T. We use the same as K so be careful!!!
    Smem_tile_kt smem_kt(&smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::Smem_tile_q::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for dO.
    Gmem_tile_do gmem_do(params.do_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                         params.d, binfo, tidx, true);
    // Allocate the shared memory tile loader for dO.
    Smem_tile_do smem_do(&smem_[0], tidx);
    Smem_tile_dot smem_dot(&smem_[0], tidx);
    // Allocate the shared memory tile loader for Q^T.
    // TODO: assert that this points to the same memory as gemm_q_k.smem_q
    Smem_tile_qt smem_qt(&smem_[Smem_tile_do::BYTES_PER_TILE], tidx);

    Smem_tile_st smem_s(&smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O + Smem_tile_dq::BYTES_PER_TILE], tidx);
    Smem_tile_st smem_dp(&smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O + Smem_tile_dq::BYTES_PER_TILE + Smem_tile_st::BYTES_PER_TILE], tidx);

    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                       params.d, binfo, tidx, true);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_dq smem_dq(&smem_[Smem_tile_do::BYTES_PER_TILE + Gemm1::SMEM_OFFSET_O], tidx);

    Gmem_softmax_sum gmem_softmax_lse(params.softmax_lse_ptr, params, tidx);
    Gmem_softmax_sum gmem_softmax_d(params.dsoftmax_sum, params, tidx);

    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    int begin = Is_causal ? loop_step_idx * Cta_tile_p::N / Cta_tile_p::M : 0;
    // Otherwise we'd be reading out-of-bound memory before the loop
    if (begin * Cta_tile_p::M >= binfo.actual_seqlen_q) {
        // Still need to zero out dk and dv before returning
        static_assert(Smem_tile_dk::NUM_LDS == Smem_tile_dv::NUM_LDS);
        uint4 dkv_out[Smem_tile_dk::NUM_LDS];
        #pragma unroll
        for (int i = 0; i < Smem_tile_dk::NUM_LDS; ++i) { dkv_out[i] = make_uint4(0u, 0u, 0u, 0u); }
        Gmem_tile_dk gmem_dk(params.dk_ptr, params.dk_row_stride_in_elts, params.dk_head_stride_in_elts,
                            params.d, binfo, tidx, false);
        if (!Is_first) { gmem_dk.move(loop_step_idx); }
        gmem_dk.store(dkv_out);
        Gmem_tile_dv gmem_dv(params.dv_ptr, params.dv_row_stride_in_elts, params.dv_head_stride_in_elts,
                            params.d, binfo, tidx, false);
        if (!Is_first) { gmem_dv.move(loop_step_idx); }
        gmem_dv.store(dkv_out);
        return;
    }

    const int steps = (params.seqlen_q + Cta_tile_p::M - 1) / Cta_tile_p::M - begin;
    // Wind gmem tiles to the correct position.
    gmem_q.move(begin);
    gmem_do.move(begin);
    gmem_o.move(begin);
    if (!Seq_parallel) { gmem_dq.move(begin); }  // If Seq_parallel, we're not using gmem_dq at all
    gmem_dq_tmp.move(begin);
    // TODO: need to move gmem_s if we want the intermediate result for debugging
    gmem_softmax_lse.move(begin);
    gmem_softmax_d.move(begin);

    if (!Is_first) {
        gmem_k.move(loop_step_idx);
        gmem_v.move(loop_step_idx);
    }

    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for V.
    gmem_v.load();
    // Trigger the loads for dO.
    gmem_do.load();
    // Trigger the loads for O.
    if (Is_first) { gmem_o.load(); }

    float p_lse[Mma_tile_p::MMAS_M * 2];
    gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_lse));

    if (!Is_first) { __syncthreads(); }
    // Commit the data for Q, dO, and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_do.commit(smem_do);
    if (Is_first) {
        dot_do_o<Gmem_tile_do::ROWS, Gmem_tile_do::THREADS_PER_ROW, elem_type>(
            gmem_do.fetch_, gmem_o.fetch_, params.p_dropout, gmem_softmax_d, tidx
        );
    }

    // // Instead of scaling dP by rp_dropout, we scale V instead
    // if (Is_dropout) {
    //     const uint32_t scale_dropout = params.scale_dropout;
    //     #pragma unroll
    //     for(int it=0; it < Gmem_tile_v::LDGS; it++){
    //         gmem_v.fetch_[it] = fmha::hmul8(scale_dropout, gmem_v.fetch_[it]);
    //     }
    // }

    gmem_v.commit(smem_v);

    // const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    // #pragma unroll
    // for(int it=0; it < Gmem_tile_k::LDGS; it++){
    //     gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    // }

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    gemm_q_k.load_q();

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Kernel_traits::V_IN_REGS ? Mma_tile_p::MMAS_K : 2][Mma_tile_p::MMAS_N];
    if (Kernel_traits::V_IN_REGS) {
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            smem_v.load(frag_v[ki], ki);
        }
    }

    float dp_sum[Mma_tile_p::MMAS_M * 2];
    gmem_softmax_d.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(dp_sum));

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_k.commit(gemm_q_k.smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for K.
    gemm_q_k.load_k();
    // Load the fragments for K^T.
    // typename Smem_tile_kt::Fragment frag_kt[2][Mma_tile_dq::MMAS_N];
    // smem_kt.load(frag_kt[0], 0);
    // typename Smem_tile_kt::Fragment frag_kt[Mma_tile_dq::MMAS_K][Mma_tile_dq::MMAS_N];
    // #pragma unroll
    // for( int ki = 0; ki < Mma_tile_dq::MMAS_K; ++ki ) {
    //     smem_kt.load(frag_kt[ki], ki);
    // }

    // Create the object to do the softmax.
    // We won't be using the shared memory for this softmax at all
    Softmax softmax(params, smem_, tidx);

    // Declare the accumulators for the 3rd gemm.
    fmha::Fragment_accumulator acc_dv[Mma_tile_dkv::MMAS_M][Mma_tile_dkv::MMAS_N];
    fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_dkv::WARPS_K>::apply(acc_dv);
    fmha::Fragment_accumulator acc_dk[Mma_tile_dkv::MMAS_M][Mma_tile_dkv::MMAS_N];
    fmha::Clear_accumulator<fmha::Accumulator_type, Cta_tile_dkv::WARPS_K>::apply(acc_dk);

    // Load over the entire sequence length.
    for (int l = 0; l < steps; l++) {
        if ((begin + l) * Cta_tile_p::M  >= binfo.actual_seqlen_q)
            break;

        // Load the fragments for V.
        // typename Smem_tile_v::Fragment frag_v[2][Mma_tile_p::MMAS_N];
        if (!Kernel_traits::V_IN_REGS) { smem_v.load(frag_v[0], 0); }

        // Load the fragments for dO.
        typename Smem_tile_do::Fragment frag_do[2][Mma_tile_p::MMAS_M];
        smem_do.load(frag_do[0], 0);

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P^T = (Q * K^T)^T.
        gemm_q_k(acc_p);

        // Load the mask for that iteration.
        mask.load(begin + l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);
        // Apply the mask.
        softmax.apply_mask(mask);
        // Scale by log-sum-exp of the softmax
        // softmax.apply_exp(p_lse);
        softmax.template scale_apply_exp</*scale_max=*/false>(p_lse, params.scale_bmm1f);
        if (Is_dropout) {
            // softmax.apply_dropout(ph, params.p_dropout_in_uint);
            // softmax.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(ph, params.p_dropout_in_uint);
            // softmax.template apply_dropout_16bits</*encode_dropout_in_sign_bit=*/true>(ph, params.p_dropout_in_uint16_t);
            unsigned int warp_idx = threadIdx.x / 32;
            // TODO: this should change after we rearrange the warps (e.g. cutlass branch)
            unsigned int block_col_idx = loop_step_idx * Cta_tile_p::N / 16 + warp_idx;
            unsigned long long philox_subsequence = (begin + l) * (binfo.actual_seqlen_k / 16) + block_col_idx;
            softmax.template apply_dropout_16bits</*encode_dropout_in_sign_bit=*/true>(ph, params.p_dropout_in_uint16_t, philox_subsequence);
        }

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_dq::MMAS_K][Mma_tile_dq::MMAS_M];
        static_assert(Mma_tile_dq::MMAS_M == Mma_tile_p::MMAS_M);
        static_assert(Mma_tile_dq::MMAS_K == Mma_tile_p::MMAS_N);
        softmax.template pack<elem_type>(frag_p);

        // Store s * dmask to smem for transpose
        smem_s.store(frag_p);

        // Trigger the load for the next Q values.
        if (l + 1 < steps) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move();
            gmem_q.load();
        }

        // if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
        //     // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
        //     __syncthreads();
        // }

        fmha::Fragment_accumulator acc_dp[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        #pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < Mma_tile_p::MMAS_N; ++ni) {
                #pragma unroll
                for (int ii = 0; ii < 8; ++ii) {
                    acc_dp[mi][ni].elt(ii) = -dp_sum[mi * 2 + ((ii / 2) % 2)];
                }
            }
        }

        // Do this part of dP^T = (dO * V^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of dO values.
            smem_do.load(frag_do[ki & 1], ki);
            if (!Kernel_traits::V_IN_REGS) {
                smem_v.load(frag_v[ki & 1], ki);
                fmha::gemm_cl<elem_type>(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1) & 1]);
            } else {
                fmha::gemm_cl<elem_type>(acc_dp, frag_do[(ki - 1) & 1], frag_v[ki - 1]);
            }
            // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l < 4))  {
            //     float2 tmp = __half22float2(reinterpret_cast<__half2 &>(frag_do[(ki - 1) & 1]));
            //     printf("frag_do=%.6f, %.6f\n", tmp.x, tmp.y);
            //     tmp = __half22float2(reinterpret_cast<__half2 &>(frag_v[(ki - 1) & 1]));
            //     printf("frag_v=%.6f, %.6f\n", tmp.x, tmp.y);
            // }
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            if (!Kernel_traits::V_IN_REGS) {
                fmha::gemm_cl<elem_type>(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1) & 1]);
            } else {
                fmha::gemm_cl<elem_type>(acc_dp, frag_do[(ki - 1) & 1], frag_v[(ki - 1)]);
            }
        }

        auto pointwise_mult = [](float p, float dp, float d) {
            return p * ((!Is_dropout) || p >= 0.f ? dp : d);
        };
        #pragma unroll
        for (int mi = 0; mi < Mma_tile_p::MMAS_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < Mma_tile_p::MMAS_N; ni++) {
                softmax.elt_[2 * mi + 0][4 * ni + 0] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 0], acc_dp[mi][ni].elt(0), dp_sum[2 * mi + 0]);
                softmax.elt_[2 * mi + 0][4 * ni + 1] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 1], acc_dp[mi][ni].elt(1), dp_sum[2 * mi + 0]);
                softmax.elt_[2 * mi + 0][4 * ni + 2] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 2], acc_dp[mi][ni].elt(4), dp_sum[2 * mi + 0]);
                softmax.elt_[2 * mi + 0][4 * ni + 3] = pointwise_mult(softmax.elt_[2 * mi + 0][4 * ni + 3], acc_dp[mi][ni].elt(5), dp_sum[2 * mi + 0]);
                softmax.elt_[2 * mi + 1][4 * ni + 0] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 0], acc_dp[mi][ni].elt(2), dp_sum[2 * mi + 1]);
                softmax.elt_[2 * mi + 1][4 * ni + 1] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 1], acc_dp[mi][ni].elt(3), dp_sum[2 * mi + 1]);
                softmax.elt_[2 * mi + 1][4 * ni + 2] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 2], acc_dp[mi][ni].elt(6), dp_sum[2 * mi + 1]);
                softmax.elt_[2 * mi + 1][4 * ni + 3] = pointwise_mult(softmax.elt_[2 * mi + 1][4 * ni + 3], acc_dp[mi][ni].elt(7), dp_sum[2 * mi + 1]);
            }
        }

        // Load the fragments for K^T.
        typename Smem_tile_kt::Fragment frag_kt[2][Mma_tile_dq::MMAS_N];
        smem_kt.load(frag_kt[0], 0);

        // Trigger the load for the next dO values.
        if (l + 1 < steps) {
            smem_do.move_to_next_write_buffer();
            gmem_do.move();
            gmem_do.load();
            if (Is_first) {
                gmem_o.move();
                gmem_o.load();
            }
        }

        softmax.template pack<elem_type>(frag_p);

        // Store dp to smem for transpose
        smem_dp.store(frag_p);

        // gmem_s.store(frag_p, mask);
        // gmem_s.move();

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_dq[Mma_tile_dq::MMAS_M][Mma_tile_dq::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_dq::WARPS_K>::apply(acc_dq);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dq::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_kt.load(frag_kt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1) & 1]);
            // fmha::gemm_cl<elem_type>(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_dq::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1) & 1]);
            // fmha::gemm_cl<elem_type>(acc_dq, frag_p[ki - 1], frag_kt[(ki - 1)]);
        }

        static_assert(Gmem_tile_dq::LOOPS == 1);

        // Swizzle the elements and do the final reduction.
        // Need to syncthreads here, otherwise the smem_dq reads from the previous iteration
        // might happen after the smem_dq writes in this iteration.
        __syncthreads();
        smem_dq.store(acc_dq, 0);

        typename Smem_tile_dot::Fragment frag_dot[2][Mma_tile_dkv::MMAS_N];
        static_assert(Smem_tile_dot::Fragment::NUM_REGS == 4);
        static_assert(Mma_tile_dkv::MMAS_K == 1);
        smem_dot.load(frag_dot[0], 0);

        // Threads in a warp is communicating via shared memory (smem_s and smem_dp)
        __syncwarp();
        typename Smem_tile_st::Fragment frag_s[Mma_tile_dkv::MMAS_K][Mma_tile_dkv::MMAS_M];
        smem_s.load(frag_s);

        if (Is_dropout) {
            #pragma unroll
            for( int ki = 0; ki < Mma_tile_dkv::MMAS_K; ki++ ) {
                #pragma unroll
                for( int mi = 0; mi < Mma_tile_dkv::MMAS_M; mi++ ) {
                    frag_s[ki][mi].template hrelu_<elem_type>();
                }
            }
        }

        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dkv::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_dot.load(frag_dot[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_dv, frag_s[(ki - 1)], frag_dot[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dkv::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_dv, frag_s[(ki - 1)], frag_dot[(ki - 1) & 1]);
        }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
        //     float2 tmp0 = __half22float2(reinterpret_cast<__half2 &>(frag_dot[0][0]));
        //     printf("frag_dot[0][0]=%.6f, %.6f\n", tmp0.x, tmp0.y);
        //     float2 tmp1 = __half22float2(reinterpret_cast<__half2 &>(frag_dot[0][1]));
        //     printf("frag_dot[0][1]=%.6f, %.6f\n", tmp1.x, tmp1.y);
        // }

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
        //     printf("l = %d, acc_dv[0][0]=%.6f, %.6f\n", l, acc_dv[0][0].elt(2), acc_dv[0][0].elt(3));
        //     printf("l = %d, acc_dv[0][1]=%.6f, %.6f\n", l, acc_dv[0][1].elt(2), acc_dv[0][1].elt(3));
        // }
        // __syncthreads();
        // Commit the values for Q and dO into shared memory.
        if (l + 1 < steps) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

        uint4 dq_out[Gmem_tile_dq::STGS_PER_LOOP];
        if (!Is_first && !Seq_parallel) { gmem_dq_tmp.load(dq_out, 0); }

        // __syncthreads();
        // Commit the values for Q and dO into shared memory.
        if (l + 1 < steps) {
            gmem_do.commit(smem_do);
            gmem_softmax_d.move();
            if (Is_first) {
                dot_do_o<Gmem_tile_do::ROWS, Gmem_tile_do::THREADS_PER_ROW, elem_type>(
                    gmem_do.fetch_, gmem_o.fetch_, params.p_dropout, gmem_softmax_d, tidx
                );
            }
            gmem_softmax_lse.move();
            gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_lse));
        }

        typename Smem_tile_st::Fragment frag_dpt[Mma_tile_dkv::MMAS_K][Mma_tile_dkv::MMAS_M];
        smem_dp.load(frag_dpt);

        gemm_q_k.reload_k();

        typename Smem_tile_qt::Fragment frag_qt[2][Mma_tile_dkv::MMAS_N];
        static_assert(Smem_tile_qt::Fragment::NUM_REGS == 4);
        static_assert(Mma_tile_dkv::MMAS_K == 1);
        smem_qt.load(frag_qt[0], 0);

        #pragma unroll
        for( int ki = 1; ki < Mma_tile_dkv::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            smem_qt.load(frag_qt[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_dk, frag_dpt[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Do the final stage of math.
        {
            int ki = Mma_tile_dkv::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_dk, frag_dpt[(ki - 1)], frag_qt[(ki - 1) & 1]);
        }

        // Make sure dQ is in shared memory.
        __syncthreads();

        if (l + 1 < steps) {
            gmem_softmax_d.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(dp_sum));
        }

        // Load from shared memory.
        smem_dq.template load</*zero_init=*/Is_first || Seq_parallel>(dq_out);

        if (!Seq_parallel) {
            const bool is_final_write =
                Is_last
                || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
                || ((Is_causal) && ((begin + l) * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
            if (is_final_write) {
                // if (Is_dropout) {
                //     dq_out[0] = fmha::fmul4(dq_out[0], params.rp_dropout);
                // }
                for (int jj = 0; jj < Gmem_tile_dq::STGS_PER_LOOP; ++jj) {
                    // dq_out[jj] = fmha::fmul4(dq_out[jj], params.scale_bmm1f);
                    dq_out[jj] = fmha::fmul4(dq_out[jj], params.scale_bmm1_rp_dropout);
                }
                // Output the values.
                gmem_dq.template store<elem_type>(dq_out, 0);
                // Move to the next part of the output.
                gmem_dq.move();
                // TODO: for parallel, need to deal with the dropout scaling
            } else  {
                // Output the values.
                gmem_dq_tmp.store(dq_out, 0);
            }
        } else {
            // We always scale dq_out before writing in this case, since we don't want to
            // have to scale at the end when copying from dq_tmp to dq.
            for (int jj = 0; jj < Gmem_tile_dq::STGS_PER_LOOP; ++jj) {
                // dq_out[jj] = fmha::fmul4(dq_out[jj], params.scale_bmm1f);
                dq_out[jj] = fmha::fmul4(dq_out[jj], params.scale_bmm1_rp_dropout);
            }
            gmem_dq_tmp.atomic_add(dq_out, 0);
        }

        // Move to the next part of the output.
        if (!(Is_first && Is_last)) { gmem_dq_tmp.move(); }

        // // Make sure the data is in shared memory.
        // __syncthreads();

        // Commit the values for Q and dO into shared memory.
        if (l + 1 < steps) {
            gemm_q_k.smem_q.move_to_next_read_buffer();
            gemm_q_k.reload_q();
            smem_qt.move_to_next_read_buffer();
            // smem_qt.load(frag_qt[0], 0);
            smem_do.move_to_next_read_buffer();
            smem_dot.move_to_next_read_buffer();
            // smem_dot.load(frag_dot[0], 0);
        }

    }  // Outer loop over the sequence length.

    if (Is_dropout) {
        for( int mi = 0; mi < Mma_tile_dkv::MMAS_M; mi++ ) {
            for( int ni = 0; ni < Mma_tile_dkv::MMAS_N; ni++ ) {
                acc_dv[mi][ni].mul_(params.rp_dropout);
            }
        }
    }
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
    //     printf("l final, acc_dv[0][0]=%.6f, %.6f\n", acc_dv[0][0].elt(2), acc_dv[0][0].elt(3));
    //     printf("l final, acc_dv[0][1]=%.6f, %.6f\n", acc_dv[0][1].elt(2), acc_dv[0][1].elt(3));
    // }
    for( int mi = 0; mi < Mma_tile_dkv::MMAS_M; mi++ ) {
        for( int ni = 0; ni < Mma_tile_dkv::MMAS_N; ni++ ) {
            // acc_dk[mi][ni].mul_(Is_dropout ? params.rp_dropout * params.scale_bmm1f : params.scale_bmm1f);
            // acc_dk[mi][ni].mul_(params.scale_bmm1f);
            acc_dk[mi][ni].mul_(params.scale_bmm1_rp_dropout);
        }
    }
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0))  {
    //     printf("l final, acc_dk=%.6f, %.6f\n", acc_dk[0][0].elt(0), acc_dk[0][0].elt(1));
    // }

    __syncthreads();
    // TODO [TD - 2022-05-04]: Are there cases where the shared mem for dV and dK are larger than
    // the total amount of shared mem?
    // Epilogue swizzle for dV
    Smem_tile_dv smem_dv(&smem_[0], tidx);
    smem_dv.template store<elem_type>(acc_dv);

    // Epilogue swizzle for dK
    Smem_tile_dk smem_dk(&smem_[Smem_tile_dv::BYTES_PER_TILE], tidx);
    smem_dk.template store<elem_type>(acc_dk);

    __syncthreads();
    uint4 dv_out[Smem_tile_dv::NUM_LDS];
    smem_dv.load(dv_out);
    Gmem_tile_dv gmem_dv(params.dv_ptr, params.dv_row_stride_in_elts, params.dv_head_stride_in_elts,
                         params.d, binfo, tidx, false);
    if (!Is_first) {
        gmem_dv.move(loop_step_idx);
    }
    gmem_dv.store(dv_out);

    uint4 dk_out[Smem_tile_dk::NUM_LDS];
    smem_dk.load(dk_out);
    Gmem_tile_dk gmem_dk(params.dk_ptr, params.dk_row_stride_in_elts, params.dk_head_stride_in_elts,
                         params.d, binfo, tidx, false);
    if (!Is_first) {
        gmem_dk.move(loop_step_idx);
    }
    gmem_dk.store(dk_out);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// loop_steps = -1 means the number of steps will be params.seqlen_k / Kernel_traits::Cta_tile_p::N.
// This template parameter is there so we can specialize with loop_steps == 1 and loop_steps == 2.
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, int loop_steps=-1, typename Params>
inline __device__ void compute_dq_dk_dv_1xN(const Params &params) {
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), 0,  std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32);

    if (loop_steps == 1) {
        compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, true, true>(params, ph, 0);
    } else if (loop_steps == 2) {
        compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, true, false>(params, ph, 0);
        compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, false, true>(params, ph, 1);
    } else {
        if (params.seqlen_k == blocksize_c) {
            compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, true, true>(params, ph, 0);
        } else {
            const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
            compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, true, false>(params, ph, 0);
            for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
                compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, false, false>(params, ph, loop_step_idx);
            }
            compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, false, true>(params, ph, max_loop_steps - 1);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, typename Params>
inline __device__ void compute_dq_dk_dv_seqparallel(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32);

    int loop_step_idx = blockIdx.z;
    compute_dq_dk_dv_1xN_one_iter<Kernel_traits, Is_dropout, Is_causal, false, false, /*Seq_parallel=*/true>(params, ph, loop_step_idx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
