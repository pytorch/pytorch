/******************************************************************************
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

#include <cmath>
#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float apply_exp_(float x, float max) {
    return __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float apply_exp2_(float x, float max) {
    return exp2f(x - max);
    // With fast-math, this produces the same PTX instruction as the assembly below
    // float diff = x - max;
    // float res;
    // asm ("ex2.approx.ftz.f32 %0, %1;\n\t" : "=f"(res) : "f"(diff));
    // return res;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int COLS> struct ReadType {};
template<> struct ReadType<4> { using T = float;};
template<> struct ReadType<8> { using T = float2;};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cta_tile, typename Kernel_traits>
struct Smem_tile_reduce {
    // Helper class to distribute MMA tiles reduced over rows per warp over quads.

    // The Mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    static constexpr int MMAS_M = Mma_tile::MMAS_M;
    static constexpr int MMAS_N = Mma_tile::MMAS_N;

    static constexpr int WARPS_M = Cta_tile::WARPS_M;
    static constexpr int WARPS_N = Cta_tile::WARPS_N;


    static constexpr int ROWS = WARPS_M * MMAS_M * 16;
    static constexpr int COLS = WARPS_N;
    static_assert(COLS == 4 || COLS == 8, "");
    static constexpr int ROWS_PER_XOR_PATTERN = (COLS == 8) ? 4 : 8;
    static constexpr int BYTES_PER_TILE = ROWS * COLS * sizeof(float);
    static constexpr int ELTS_PER_TILE = ROWS * COLS;

    using read_t = typename ReadType<COLS>::T;

    __device__ inline Smem_tile_reduce(float *smem_, const int tidx) {

        int lane = tidx % 32;
        int warp = tidx / 32;

        int warp_m = warp % WARPS_M;
        int warp_n = warp / WARPS_M;

        qid_ = lane % 4;
        int qp = lane / 4;

        // Swizzle the column to avoid 2-fold bank conflicts when we have 8 warps.
        // This won't affect reading as we assume commutative reduction ops.
        const int col = warp_n ^ (qp / ROWS_PER_XOR_PATTERN);
        smem_write_ = &smem_[warp_m * 16 * MMAS_M * WARPS_N + qp * WARPS_N + col];
        smem_read_ = &reinterpret_cast<read_t *>(smem_)[warp_m * 16 * MMAS_M * 4 + qp * 4 + qid_];
        smem_read_row_ = &reinterpret_cast<read_t *>(smem_)[warp_m * 16 * MMAS_M * 4 + qid_];

    }

    __device__ inline void store(float (&frag)[2 * MMAS_M]) {
        if( qid_ == 0 ) {
            #pragma unroll
            for( int mi = 0; mi < MMAS_M; mi++ ) {
                int offset = mi * 16 * WARPS_N;
                smem_write_[offset + 0 * 8 * WARPS_N] = frag[mi * 2 + 0];
                smem_write_[offset + 1 * 8 * WARPS_N] = frag[mi * 2 + 1];
            }
        }
    }

    __device__ inline void load(read_t (&frag)[2 * MMAS_M]) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            int offset = mi * 16 * 4;
            frag[mi * 2 + 0] = smem_read_[offset + 0 * 8 * 4];
            frag[mi * 2 + 1] = smem_read_[offset + 1 * 8 * 4];
        }
    }

    __device__ inline void load_row(read_t (&frag)[MMAS_M], int row) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            int offset = mi * 16 * 4;
            frag[mi] = smem_read_row_[offset + 0 * 8 * 4 + row * 4];
        }
    }

    int qid_;
    float *smem_write_;
    read_t *smem_read_;
    read_t *smem_read_row_;

};


template<typename Cta_tile, typename Kernel_traits>
struct Softmax_base {

    // The Mma tile.
    using Mma_tile = fmha::Hmma_tile<Cta_tile>;

    // The number of MMAs in M/N dimensions.
    static constexpr int MMAS_M = Mma_tile::MMAS_M;
    static constexpr int MMAS_N = Mma_tile::MMAS_N;

    // The number of groups of warp such that we have at most 4 warps writing consecutive elements.
    static constexpr int GROUPS = fmha::DivUpConstexpr(Cta_tile::WARPS_N, 4);
    // The number of elements that we are going to store per row.
    static constexpr int ELEMENTS_PER_ROW = Cta_tile::WARPS_N / GROUPS;
    // The number of rows.
    static constexpr int ROWS = Cta_tile::M * GROUPS;
    // The total number of elements.
    static constexpr int ELEMENTS = ROWS * ELEMENTS_PER_ROW;

    // Ctor.
    template<typename Params>
    inline __device__ Softmax_base(const Params &params, void *smem, int tidx)
        :  // packed_mask_ptr_(reinterpret_cast<const char*>(params.packed_mask_ptr)),
          smem_(reinterpret_cast<float *>(smem)), tidx_(tidx) {

        // Extract the position in the warp.
        int warp = tidx / Cta_tile::THREADS_PER_WARP;
        int lane = tidx % Cta_tile::THREADS_PER_WARP;

        // Decompose the warp index into M and N.
        int warp_m = warp % Cta_tile::WARPS_M;
        int warp_n = warp / Cta_tile::WARPS_M;

        // Decompose the warp-n index into group/position-inside-the-group.
        int warp_g = warp_n / ELEMENTS_PER_ROW;
        int warp_i = warp_n % ELEMENTS_PER_ROW;

        // The location written by the threads.
        int write_row = warp_g * (ROWS / GROUPS) + warp_m * Mma_tile::M_PER_MMA + lane / 4;
        int write_col = warp_i;

        // Assemble the write pointer.
        smem_write_ = &smem_[write_row * ELEMENTS_PER_ROW + write_col];

        // Assemble the read pointer.
        smem_read_ = &smem_[warp_m * Mma_tile::M_PER_MMA + lane / 4];
    }

    template<bool zero=false, typename Mask>
    inline __device__ void apply_mask(const Mask &mask) {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ii = 0; ii < 2; ++ii ) {
                #pragma unroll
                for( int ni = 0; ni < MMAS_N; ++ni ) {
                    #pragma unroll
                    for( int jj = 0; jj < 4; ++jj ) {
                        if( !mask.is_valid(mi, ni, ii, jj) ) {
                            elt_[2 * mi + ii][4 * ni + jj] = zero ? 0.f : -INFINITY;
                        }
                    }
                }
            }
        }
    }

    // Apply the exp to all the elements.
    template <bool scale_max=true>
    inline __device__ void scale_apply_exp(const float (&max)[MMAS_M * 2], const float scale_) {
        const float max_scale = scale_max ? scale_ * M_LOG2E : M_LOG2E;
        const float scale = scale_ * M_LOG2E;
        #pragma unroll
        for( int mi = 0; mi < MMAS_M * 2; ++mi ) {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            const float max_scaled = max[mi] * max_scale;
            #pragma unroll
            for( int ni = 0; ni < MMAS_N * 4; ++ni ) {
                elt_[mi][ni] = apply_exp2_(elt_[mi][ni] * scale, max_scaled);
            }
        }
    }

    template <bool encode_dropout_in_sign_bit=false>
    inline __device__ void apply_dropout_16bits(Philox &ph, uint16_t p_dropout_in_uint16_t) {
        // We encode the dropout pattern in the sign bit of the non-negative
        // softmax to distinguish from pre-existing zeros
        auto encode_dropout = [](bool keep, float val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : float(0));
        };
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ni++ ) {
                uint4 random_uint4 = ph();
                uint16_t (&rnd)[8] = reinterpret_cast<uint16_t (&)[8]>(random_uint4);
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("ni = %d, ph  Philox: %u, %u, %u, %u\n", ni, rnd.x, rnd.y, rnd.z, rnd.w);
                // }
                #pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        elt_[mi * 2 + ii][4 * ni + jj] =
                            encode_dropout(rnd[ii * 4 + jj] <= p_dropout_in_uint16_t, elt_[mi * 2 + ii][4 * ni + jj]);
                    }
                }
            }
        }
    }

    template <bool encode_dropout_in_sign_bit=false>
    inline __device__ void apply_dropout_16bits(Philox &ph0, Philox &ph1, uint16_t p_dropout_in_uint16_t) {
        // We encode the dropout pattern in the sign bit of the non-negative
        // softmax to distinguish from pre-existing zeros
        auto encode_dropout = [](bool keep, float val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : float(0));
        };
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; mi++ ) {
            static_assert(MMAS_N % 2 == 0, "");
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ni += 2 ) {
                uint4 random_uint4 = ph0();
                uint16_t (&rnd0)[8] = reinterpret_cast<uint16_t (&)[8]>(random_uint4);
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("ni = %d, ph  Philox: %u, %u, %u, %u\n", ni, rnd0.x, rnd0.y, rnd0.z, rnd0.w);
                // }
                #pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        elt_[mi * 2 + ii][4 * ni + jj] =
                            encode_dropout(rnd0[ii * 4 + jj] <= p_dropout_in_uint16_t, elt_[mi * 2 + ii][4 * ni + jj]);
                    }
                }
                random_uint4 = ph1();
                uint16_t (&rnd1)[8] = reinterpret_cast<uint16_t (&)[8]>(random_uint4);
                // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                //     printf("ni = %d, ph  Philox: %u, %u, %u, %u\n", ni, rnd1.x, rnd1.y, rnd1.z, rnd1.w);
                // }
                #pragma unroll
                for (int ii = 0; ii < 2; ++ii) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        elt_[mi * 2 + ii][4 * (ni + 1) + jj] =
                            encode_dropout(rnd1[ii * 4 + jj] <= p_dropout_in_uint16_t, elt_[mi * 2 + ii][4 * (ni + 1) + jj]);
                    }
                }
            }
        }
    }

    // Shared memory for the CTA-wide reduction.
    float *smem_, *smem_write_, *smem_read_;
    // The current thread index.
    int tidx_;
    // The elements.
    float elt_[MMAS_M * 2][MMAS_N * 4];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile, typename Kernel_traits>
struct Softmax : public Softmax_base<Cta_tile, Kernel_traits> {

    // The base class.
    using Base = Softmax_base<Cta_tile, Kernel_traits>;

    static constexpr int WARPS_M = Cta_tile::WARPS_M;
    static constexpr int WARPS_N = Cta_tile::WARPS_N;
    // The MMAs.
    static constexpr int MMAS_M = Base::MMAS_M;
    static constexpr int MMAS_N = Base::MMAS_N;

    using Smem_tile_red = Smem_tile_reduce<Cta_tile, Kernel_traits>;
    static_assert(Smem_tile_red::ELTS_PER_TILE == Cta_tile::M * WARPS_N, "");
    // Ctor.
    template<typename Params>
    inline __device__ Softmax(const Params &params, void *smem, int tidx)
        : Base(params, smem, tidx)
        , smem_sum_(static_cast<float*>(smem), tidx)
        , smem_max_(static_cast<float*>(smem) + Smem_tile_red::ELTS_PER_TILE, tidx) {
    }

    // Pack the data to a fragment for the next GEMM.
    inline __device__ void pack_noconvert(cutlass::Array<float, MMAS_M * MMAS_N * 8> &frag) const {
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ki = 0; ki < MMAS_N; ++ki ) {
                // 1st row - 4 elements per row.
                frag[ki * MMAS_M * 8 + mi * 8 + 0] = this->elt_[2 * mi + 0][4 * ki + 0];
                frag[ki * MMAS_M * 8 + mi * 8 + 1] = this->elt_[2 * mi + 0][4 * ki + 1];
                frag[ki * MMAS_M * 8 + mi * 8 + 4] = this->elt_[2 * mi + 0][4 * ki + 2];
                frag[ki * MMAS_M * 8 + mi * 8 + 5] = this->elt_[2 * mi + 0][4 * ki + 3];
                // 2nd row - 4 elements per row.
                frag[ki * MMAS_M * 8 + mi * 8 + 2] = this->elt_[2 * mi + 1][4 * ki + 0];
                frag[ki * MMAS_M * 8 + mi * 8 + 3] = this->elt_[2 * mi + 1][4 * ki + 1];
                frag[ki * MMAS_M * 8 + mi * 8 + 6] = this->elt_[2 * mi + 1][4 * ki + 2];
                frag[ki * MMAS_M * 8 + mi * 8 + 7] = this->elt_[2 * mi + 1][4 * ki + 3];
            }
        }
    }

    template <typename FragmentC>
    inline __device__ void unpack_noscale(const FragmentC (&acc)) {
        static_assert(FragmentC::kElements == MMAS_M * MMAS_N * 8, "");
        #pragma unroll
        for( int mi = 0; mi < MMAS_M; ++mi ) {
            #pragma unroll
            for( int ni = 0; ni < MMAS_N; ++ni ) {
                // 1st row - 4 elements per row.
                this->elt_[2 * mi + 0][4 * ni + 0] = acc[mi * MMAS_N * 8 + ni * 8 + 0];
                this->elt_[2 * mi + 0][4 * ni + 1] = acc[mi * MMAS_N * 8 + ni * 8 + 1];
                this->elt_[2 * mi + 0][4 * ni + 2] = acc[mi * MMAS_N * 8 + ni * 8 + 4];
                this->elt_[2 * mi + 0][4 * ni + 3] = acc[mi * MMAS_N * 8 + ni * 8 + 5];
                // 2nd row - 4 elements per row.
                this->elt_[2 * mi + 1][4 * ni + 0] = acc[mi * MMAS_N * 8 + ni * 8 + 2];
                this->elt_[2 * mi + 1][4 * ni + 1] = acc[mi * MMAS_N * 8 + ni * 8 + 3];
                this->elt_[2 * mi + 1][4 * ni + 2] = acc[mi * MMAS_N * 8 + ni * 8 + 6];
                this->elt_[2 * mi + 1][4 * ni + 3] = acc[mi * MMAS_N * 8 + ni * 8 + 7];
            }
        }
    }

    template<bool zero_init=true, typename Operator>
    __device__ inline void thread_reduce_(float (&frag)[2 * MMAS_M], Operator &op) {
        #pragma unroll
        for( int mi = 0; mi < 2 * MMAS_M; mi++ ) {
            frag[mi] = zero_init ? this->elt_[mi][0] : op(frag[mi], this->elt_[mi][0]);
            #pragma unroll
            for( int ni = 1; ni < 4 * MMAS_N; ni++ ) {
                frag[mi] = op(frag[mi], this->elt_[mi][ni]);
            }
        }
    }

    template<bool zero_init=true, typename Operator>
    __device__ inline void reduce_(float (&frag)[2 * MMAS_M], Operator &op, Smem_tile_red & smem_red) {
        thread_reduce_<zero_init>(frag, op);
        quad_reduce(frag, frag, op);
        smem_red.store(frag);
        __syncthreads();
        typename Smem_tile_red::read_t tmp[2 * MMAS_M];
        smem_red.load(tmp);
        quad_allreduce(frag, tmp, op);
    }

    template<bool zero_init=true>
    __device__ inline void reduce_max(float (&frag)[2 * MMAS_M]){
        MaxOp<float> max;
        reduce_<zero_init>(frag, max, smem_max_);
    }

    __device__ inline void reduce_sum(float (&frag)[2 * MMAS_M]){
        SumOp<float> sum;
        reduce_(frag, sum, smem_sum_);
    }

    template<bool zero_init=true>
    __device__ inline void reduce_sum_before_sync_(float (&frag)[2 * MMAS_M]){
        SumOp<float> sum;
        thread_reduce_<zero_init>(frag, sum);
        quad_reduce(frag, frag, sum);
        smem_sum_.store(frag);
    }

    template<int NROWS, typename Operator>
    __device__ inline void reduce_after_sync_(float (&frag)[NROWS][MMAS_M],
                                              const int (&rows)[NROWS],
                                              Operator &op, Smem_tile_red & smem_red) {
        #pragma unroll
        for (int ii = 0; ii < NROWS; ii++) {
            typename Smem_tile_red::read_t tmp[MMAS_M];
            smem_red.load_row(tmp, rows[ii]);
            quad_allreduce(frag[ii], tmp, op);
        }
    }

    template<int NROWS>
    __device__ inline void reduce_sum_after_sync_(float (&frag)[NROWS][MMAS_M],
                                                  const int (&rows)[NROWS]){
        SumOp<float> sum;
        reduce_after_sync_(frag, rows, sum, smem_sum_);
    }

    template<int NROWS>
    __device__ inline void reduce_max_after_sync_(float (&frag)[NROWS][MMAS_M],
                                                  const int (&rows)[NROWS]){
        MaxOp<float> max;
        reduce_after_sync_(frag, rows, max, smem_max_);
    }

    Smem_tile_red smem_max_;
    Smem_tile_red smem_sum_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
