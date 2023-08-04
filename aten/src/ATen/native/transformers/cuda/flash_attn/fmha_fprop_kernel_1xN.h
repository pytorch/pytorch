/***************************************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/transformers/cuda/flash_attn/fmha_kernel.h>
#include <ATen/native/transformers/cuda/flash_attn/kernel_traits.h>
#include <ATen/native/transformers/cuda/flash_attn/gemm.h>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits>
struct Gemm_Q_K_base {
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;
    using Smem_tile_q = typename Kernel_traits::Smem_tile_q;
    using Smem_tile_k = typename Kernel_traits::Smem_tile_k;
    using Fragment_q = typename Smem_tile_q::Fragment;
    using Fragment_k = typename Smem_tile_k::Fragment;

    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;

    static constexpr int SMEM_BYTES_SOFTMAX = Cta_tile_p::M * Cta_tile_p::WARPS_N * sizeof(float) * 2;

    __device__ inline Gemm_Q_K_base(char * smem_ptr_q, char * smem_ptr_k, const int tidx)
        : smem_q(smem_ptr_q, tidx)
        , smem_k(smem_ptr_k, tidx) {

    }

    __device__ inline void load_q() {
        smem_q.load(frag_q[0], 0);
    }

    __device__ inline void reload_q() {
        smem_q.load(frag_q[0], 0);
    }

    Fragment_q frag_q[2][Mma_tile_p::MMAS_M];
    Smem_tile_q smem_q;
    Smem_tile_k smem_k;
};

template<typename Kernel_traits, bool K_in_regs, typename elem_type_=__half>
struct Gemm_Q_K : public Gemm_Q_K_base<Kernel_traits> {

    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    // If V is stored in shared memory, we can't load K using the same shared memory.
    static_assert(Kernel_traits::V_IN_REGS);

    static constexpr int SMEM_OFFSET_O = Smem_tile_q::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);

    // Q | K / V
    //   | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + std::max((SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE,
                                               Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX);

    __device__ inline Gemm_Q_K(char * smem_, const int tidx)
        : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_p::MMAS_K; ++ki ) {
            Base::smem_k.load(frag_k[ki], ki);
        }
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1)]);
        }
    }

    __device__ inline void reload_k(){
        // Noop.
    }

    Fragment_k frag_k[Mma_tile_p::MMAS_K][Mma_tile_p::MMAS_N];
};


template<typename Kernel_traits, typename elem_type_>
struct Gemm_Q_K<Kernel_traits, false, elem_type_> : public Gemm_Q_K_base<Kernel_traits> {
    using Base = Gemm_Q_K_base<Kernel_traits>;
    using Smem_tile_o = typename Base::Smem_tile_o;
    using Smem_tile_q = typename Base::Smem_tile_q;
    using Smem_tile_k = typename Base::Smem_tile_k;
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;
    using Fragment_k = typename Base::Fragment_k;
    using Mma_tile_p = typename Base::Mma_tile_p;
    using elem_type = elem_type_;
    Fragment_k frag_k[2][Mma_tile_p::MMAS_N];

    static constexpr bool SHARE_SMEM_FOR_K_AND_V = Kernel_traits::SHARE_SMEM_FOR_K_AND_V;
    static constexpr bool V_IN_REGS = Kernel_traits::V_IN_REGS;
    static_assert(V_IN_REGS || !SHARE_SMEM_FOR_K_AND_V);

    static constexpr int SMEM_OFFSET_V = Smem_tile_q::BYTES_PER_TILE + (SHARE_SMEM_FOR_K_AND_V ? 0 : Smem_tile_k::BYTES_PER_TILE);
    static_assert(Smem_tile_v::BYTES_PER_TILE == (int) Smem_tile_k::BYTES_PER_TILE);
    static constexpr int SMEM_OFFSET_O = SMEM_OFFSET_V + Smem_tile_v::BYTES_PER_TILE;
    static constexpr int SMEM_OFFSET_SOFTMAX = SMEM_OFFSET_O + Smem_tile_o::BYTES_PER_TILE;

    // If V_IN_REGS and SHARE_SMEM_FOR_K_AND_V:      Q | K/V | O | SOFTMAX
    // If !V_IN_REGS (then !SHARE_SMEM_FOR_K_AND_V): Q | K   | V | O | SOFTMAX
    static constexpr int SMEM_BYTES = Smem_tile_q::BYTES_PER_TILE
                                    + (SHARE_SMEM_FOR_K_AND_V ? 1 : 2) * Smem_tile_k::BYTES_PER_TILE
                                    + Smem_tile_o::BYTES_PER_TILE + Base::SMEM_BYTES_SOFTMAX;

    __device__ inline Gemm_Q_K(char * smem_, const int tidx)
      : Base(smem_, smem_ + Smem_tile_q::BYTES_PER_TILE, tidx) {
    }

    __device__ inline void load_k(){
        Base::smem_k.load(frag_k[0], 0);
    }

    template<typename Acc, int M, int N>
    __device__ inline void operator()(Acc (&acc_p)[M][N]){
        // Do this part of P^T = (Q * K^T)^T.
        #pragma unroll
        for( int ki = 1; ki < Mma_tile_p::MMAS_K; ++ki ) {
            // Trigger the load from shared memory for the next series of Q values.
            Base::smem_q.load(Base::frag_q[ki & 1], ki);
            Base::smem_k.load(frag_k[ki & 1], ki);
            // Do the math for the values already in registers.
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
        // Do the final stage of math.
        {
            int ki = Mma_tile_p::MMAS_K;
            fmha::gemm_cl<elem_type>(acc_p, Base::frag_q[(ki - 1) & 1], frag_k[(ki - 1) & 1]);
        }
    }

    __device__ inline void reload_k(){
        Base::smem_k.load(frag_k[0], 0);
    }
};

template<typename Kernel_traits>
constexpr size_t get_dynamic_smem_size(){
    return Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>::SMEM_BYTES;
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, bool Is_first, bool Is_last, typename Params, typename Prng>
inline __device__ void device_1xN_(const Params &params, const int bidb, const int bidh, int steps, Prng &ph, const int loop_step_idx) {

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
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_o_tmp = fmha::Gmem_tile_o<Cta_tile_o, 4>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    using Smem_softmax_sum = typename Kernel_traits::Smem_dp_sum;

    using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS, elem_type>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    // How many steps to jump per iteration, which is the same as params.num_splits.
    const int step_stride = gridDim.z;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    // if( binfo.stop_early() ) return;
    if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return;

    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params.q_ptr, params.q_row_stride_in_elts, params.q_head_stride_in_elts,
                       params.d, binfo, tidx, true);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                       params.d, binfo, tidx);
    Gmem_tile_o_tmp gmem_o_tmp(params.o_tmp_ptr, params.o_tmp_row_stride_in_elts,
                               params.o_tmp_head_stride_in_elts, params.d, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    Gmem_softmax_sum gmem_softmax_lse(params.softmax_lse_ptr, params, tidx);

    // Wind gmem tiles to the correct position.
    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    int begin = Is_causal ? loop_step_idx * Cta_tile_p::N / Cta_tile_p::M : 0;
    // We want begin to be a multiple of gridDim.z
    // This is because the row indices processed by each threadblock must align between the
    // loop steps, otherwise we have a dependency between the blocks.
    // For example, threadblock with blockIdx.z == 1 must process row indices that are
    // k * gridDim.z + 1 for integer k.
    const int begin_mod_z = begin % gridDim.z;
    begin = begin_mod_z <= blockIdx.z ? begin - begin_mod_z : begin + gridDim.z - begin_mod_z;
    // Otherwise we'd be reading out-of-bound memory before the loop
    if ((begin + blockIdx.z) * Cta_tile_p::M >= binfo.actual_seqlen_q) return;
    const int steps_og = steps;
    steps -= begin;
    gmem_q.move(begin + blockIdx.z);
    gmem_o.move(begin + blockIdx.z);
    gmem_o_tmp.move(begin + blockIdx.z);
    if (Return_softmax) {
        gmem_s.move(begin + blockIdx.z);
    }
    gmem_softmax_lse.move(begin + blockIdx.z);
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("begin = %d, steps = %d\n", begin, steps);
    // }

    fmha::Mask<Cta_tile_p, Is_causal> mask(binfo, tidx, loop_step_idx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params.k_ptr, params.k_row_stride_in_elts, params.k_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params.v_ptr, params.v_row_stride_in_elts, params.v_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];

    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    if (!Is_first) {
        gmem_k.move(loop_step_idx);
        gmem_v.move(loop_step_idx);
        if (Return_softmax) { gmem_s.move(loop_step_idx * steps_og); }
    }

    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for V.
    gmem_v.load();

    if (!Is_first) { __syncthreads(); }

    float p_prev_lse[Mma_tile_p::MMAS_M * 2];
    if (!Is_first) {
        gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
    }

    // Commit the data for Q and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);

    // const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    // #pragma unroll
    // for(int it=0;it < Gmem_tile_k::LDGS;it++){
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
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
        smem_v.load(frag_v[ki], ki);
    }

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

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_SOFTMAX], tidx);

    Smem_softmax_sum smem_softmax_lse(reinterpret_cast<float *>(&smem_[Gemm1::SMEM_BYTES]), tidx);

    // Load over the entire sequence length.
    for (int l = blockIdx.z; l < steps; l += step_stride) {
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z <= 1)) {
        //     printf("l = %d\n", l);
        // }
        if ((begin + l) * Cta_tile_p::M >= binfo.actual_seqlen_q) break;

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P = Q * K^T.
        gemm_q_k(acc_p);

        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("acc_p=%.6f, %.6f\n", acc_p[0][0].elt(0), acc_p[0][0].elt(1));
        // }

        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        if (!Is_first) { gmem_o_tmp.load(out, 0); }

        // Trigger the load for the next Q values.
        if (l + step_stride < steps) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move(step_stride);
            gmem_q.load();
        }

        // Load the mask for that iteration.
        mask.load(begin + l);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l < step_stride ) {
            // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
            __syncthreads();
        }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l >= 0))  {
        //         printf("p_prev_lse=%.6f, %.6f\n", p_prev_lse[0], p_prev_lse[1]);
        //     }
        // }
        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        if (!Is_first) {
            smem_softmax_lse.store_pair(p_prev_lse);
            // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi]; }
            for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi] / params.scale_bmm1f; }
        }

        // Trigger the load for the next LSE values.
        if (l + step_stride < steps) {
            if (!Is_first) {
                gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse),
                                           step_stride);
            }
        }

        softmax.template reduce_max</*zero_init=*/Is_first>(p_max);

        // if ((threadIdx.x == 0) && (l == 38)) {
        //     printf("loop_step_idx %d, p_max = %.6f, %.6f., p_prev_lse = %.6f, %.6f\n", loop_step_idx, p_max[0], p_max[1], Is_first ? -10000.f : p_prev_lse[0], Is_first ? -10000.f : p_prev_lse[1]);
        // }

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after reduce_max=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the exponential value.
        // softmax.apply_exp(p_max);
        softmax.scale_apply_exp(p_max, params.scale_bmm1f);

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after apply_exp=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        // if (!Is_first) {
        //     int warp = tidx / Cta_tile_p::THREADS_PER_WARP;
        //     int lane = tidx % Cta_tile_p::THREADS_PER_WARP;
        //     for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) {
        //         p_sum[mi] = ((warp == 0) && (lane % 4 == 0)) ? expf(p_prev_lse[mi] - p_max[mi]) : 0;
        //     }
        // }
        // softmax.reduce_sum(p_sum);
        softmax.reduce_sum_before_sync_(p_sum);
        // softmax.template reduce_sum_before_sync_</*zero_init=*/Is_first>(p_sum);

        // float p_sum_log[Mma_tile_p::MMAS_M * 2];
        // for (int mi = 0; mi  < Mma_tile_p::MMAS_M * 2; ++mi) {
        //     float sum = p_sum[mi];
        //     // p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] + __logf(sum);
        //     constexpr float kLog2e = M_LOG2E;
        //     p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] * kLog2e + __log2f(sum);
        // }
        // // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum));
        // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum_log));
        // gmem_softmax_lse.move();

        // // Finalize softmax on the accumulators of P^T.
        // softmax.scale(p_sum);

        constexpr bool encode_dropout_in_sign_bit = Return_softmax;
        if (Is_dropout) {
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph, params.p_dropout_in_uint);
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph, ph1, params.p_dropout_in_uint);
            // softmax.template apply_dropout_16bits<encode_dropout_in_sign_bit>(ph, ph1, params.p_dropout_in_uint16_t);
            unsigned int warp_idx = threadIdx.x / 32;
            // TODO: this should change after we rearrange the warps (e.g. cutlass branch)
            unsigned int block_col_idx = loop_step_idx * Cta_tile_p::N / 16 + warp_idx;
            // We want to use actual_seqlen_k, not seqlen_k, since seqlen_k could be rounded
            // differently in the fwd and bwd pass. E.g., for d=128 on A100, fwd rounds seqlen_k
            // to multiples of 256 while bwd rounds seqlen_k to multiples of 128.
            unsigned long long philox_subsequence = (begin + l) * (binfo.actual_seqlen_k / 16) + block_col_idx;
            softmax.template apply_dropout_16bits<encode_dropout_in_sign_bit>(ph, params.p_dropout_in_uint16_t, philox_subsequence);
        }

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        static_assert(Mma_tile_o::MMAS_M == Mma_tile_p::MMAS_M);
        static_assert(Mma_tile_o::MMAS_K == Mma_tile_p::MMAS_N);
        softmax.template pack<elem_type>(frag_p);
        if (Return_softmax) {
            gmem_s.store(frag_p, mask);
            gmem_s.move(step_stride);
        }

        // Commit the values for Q into shared memory.
        if (l + step_stride < steps) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

        if (Is_dropout && encode_dropout_in_sign_bit) {
            #pragma unroll
            for( int ki = 0; ki < Mma_tile_o::MMAS_K; ki++ ) {
                #pragma unroll
                for( int mi = 0; mi < Mma_tile_o::MMAS_M; mi++ ) {
                    frag_p[ki][mi].template hrelu_<elem_type>();
                }
            }
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
            fmha::gemm_cl<elem_type>(acc_o, frag_p[ki], frag_v[ki]);
            // if ((threadIdx.x == 4) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //     float2 tmp_p = __half22float2(reinterpret_cast<__half2 &>(frag_p[ki]));
            //     float2 tmp_v = __half22float2(reinterpret_cast<__half2 &>(frag_v[ki]));
            //     printf("Per warp, threadIdx.x = %d, frag_p = %.6f, %.6f, frag_v = %.6f, %.6f, acc_o=%.6f\n", threadIdx.x, tmp_p.x, tmp_p.y, tmp_v.x, tmp_v.y, acc_o[0][0].elt(0));
            // }
        }

        // if ((threadIdx.x % 32 == 16) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //     printf("Per warp, threadIdx.x = %d, acc_o=%.6f\n", threadIdx.x, acc_o[0][2].elt(0));
        // }

        // The mapping from tidx to rows changes between the softmax and the
        // O-reduction. So we recalculate the max.
        float p_max_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        int rows[Gmem_tile_o::STGS_PER_LOOP];
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            rows[jj] = tidx / Gmem_tile_o::THREADS_PER_ROW + jj * Gmem_tile_o::ROWS_PER_STG;
        }
        softmax.reduce_max_after_sync_(p_max_o, rows);
        static_assert(Mma_tile_o::MMAS_M == 1);
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            p_max_o[jj][0] *= params.scale_bmm1f;
        }
        float p_prev_scale_o[Gmem_tile_o::STGS_PER_LOOP];
        if (!Is_first) {
            smem_softmax_lse.load(p_prev_scale_o, rows);
        }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("p_prev_scale_o=%.6f\n", p_prev_scale_o[0]);
        //     }
        // }

        static_assert(Gmem_tile_o::LOOPS == 1);

        // Swizzle the elements and do the final reduction.
        smem_o.store(acc_o, 0);

        // Make sure the data is in shared memory.
        __syncthreads();

        static_assert(Mma_tile_o::MMAS_M == 1);
        float p_sum_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        softmax.reduce_sum_after_sync_(p_sum_o, rows);
        if (!Is_first) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                p_prev_scale_o[jj] = expf(p_prev_scale_o[jj] - p_max_o[jj][0]);
                p_sum_o[jj][0] += p_prev_scale_o[jj];
            }
        }

        float p_sum_log[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            p_sum_log[jj][0] = (sum == 0.f || sum != sum) ? -INFINITY : p_max_o[jj][0] + __logf(sum);
            // if (sum == 0.f || sum != sum) {
            //     printf("loop_step_idx = %d, l = %d, tidx = %d, sum = %.6f, p_max_o = %.6f\n", loop_step_idx, l, tidx, sum, p_max_o[jj][0]);
            // }
            // if (Is_first) {
            //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //         printf("p_sum_log=%.6f\n", p_sum_log[jj][0]);
            //     }
            // }
            if (tidx % Gmem_tile_o::THREADS_PER_ROW == 0) {
                gmem_softmax_lse.store_row(
                    reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M]>(p_sum_log[jj]), rows[jj]);
            }
        }
        gmem_softmax_lse.move(step_stride);

        // Load from shared memory.
        if (!Is_first) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                out[jj] = fmha::fmul4(out[jj], p_prev_scale_o[jj]);
            }
        }
        smem_o.template load</*zero_init=*/Is_first>(out);

        const bool is_final_write =
            Is_last
            || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
            || ((Is_causal) && ((begin + l) * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if (Is_dropout && is_final_write) {
                inv_sum *= params.rp_dropout;
            }
            out[jj] = fmha::fmul4(out[jj], inv_sum);
        }

        // if (Is_dropout && Is_last) {
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         out[jj] = fmha::fmul4(out[jj], params.rp_dropout);
        //     }
        // }

        // Output the values.
        if (is_final_write) {
            gmem_o.template store<elem_type>(out, 0);
            gmem_o.move(step_stride);
        } else {
            gmem_o_tmp.store(out, 0);
        }

        // Move to the next part of the output.
        if (!(Is_first && Is_last)) { gmem_o_tmp.move(step_stride); }
        gemm_q_k.reload_k();

        // Make sure we are reading from the correct buffer.
        gemm_q_k.smem_q.move_to_next_read_buffer();
        // Trigger the load from shared memory for the next series of Q values.
        if (l + step_stride < steps) {
            gemm_q_k.reload_q();
        }
    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, typename Params>
inline __device__ void device_1xN_loop(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 16 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 16 block within the attention matrix, we can generate the exact same dropout pattern.
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    if (params.philox_args.captured_) {
        *params.seed = std::get<0>(seeds);
        *params.extragraph_offset = std::get<1>(seeds);
    }

    Philox ph(std::get<0>(seeds), 0, std::get<1>(seeds) + (bidb * params.h + bidh) * 32 + tidx % 32);
    constexpr int M = Kernel_traits::Cta_tile_p::M;
    const int STEPS = (params.seqlen_q + M - 1) / M;

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    if (params.seqlen_k == blocksize_c) {
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, true>(params, bidb, bidh, STEPS, ph, 0);
    } else {
        const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, false>(params, bidb, bidh, STEPS, ph, 0);
        for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
            fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, false>(params, bidb, bidh, STEPS, ph, loop_step_idx);
        }
        fmha::device_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, true>(params, bidb, bidh, STEPS, ph, max_loop_steps - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
