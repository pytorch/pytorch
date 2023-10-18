/******************************************************************************
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
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <ATen/native/transformers/cuda/flash_attn/philox.cuh>
#include <ATen/native/transformers/cuda/flash_attn/utils.h>

namespace pytorch_flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

template <typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const uint32_t max_seqlen_k,
                                  const uint32_t col_idx_offset_ = 0) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const uint32_t col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const uint32_t col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

template <typename Engine, typename Layout>
inline __device__ void apply_mask_causal(Tensor<Engine, Layout> &tensor, const uint32_t col_idx_offset_,
                                         const uint32_t max_seqlen_k, const uint32_t row_idx_offset_,
                                         const uint32_t warp_row_stride) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    // const uint32_t row_idx_offset = row_idx_offset_ + lane_id / 4;
    const uint32_t row_idx_offset = row_idx_offset_;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const uint32_t row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const uint32_t row_idx = row_idx_base + i * 8;
            const uint32_t col_idx_limit = std::min(max_seqlen_k, row_idx + 1);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const uint32_t col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const uint32_t col_idx = col_idx_base + j;
                    if (col_idx >= col_idx_limit) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
            // if (cute::thread0()) {
            //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i, row_idx, max_seqlen_k);
            //     print(tensor(make_coord(i, mi), _));
            //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
            // }
        }
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &idx_rowcol,
    const uint32_t col_idx_offset_, const uint32_t max_seqlen_k, const uint32_t row_idx_offset_)
{
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 2, "Only support 2D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
    CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const uint32_t col_idx_limit = std::min(max_seqlen_k, 1 + row_idx_offset_ + get<0>(idx_rowcol(mi, 0)));
        #pragma unroll
        for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
            if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
                tensor(mi, ni) = -INFINITY;
            }
        }
        // if (cute::thread0()) {
        //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j, col_idx, max_seqlen_k);
        //     print(tensor(_, make_coord(j, ni)));
        //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
        // }
    }
}

template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
inline __device__ void apply_dropout(Tensor<Engine, Layout> &tensor, uint8_t p_dropout_in_uint8_t,
                                     unsigned long long seed, unsigned long long offset,
                                     uint32_t block_row_start, uint32_t block_col_start,
                                     uint32_t block_row_stride) {
    // tensor has shape (8, MMA_M, MMA_N / 2)
    using T = typename Engine::value_type;
    auto encode_dropout = [](bool keep, T val) {
        return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
    };
    static_assert(decltype(size<2>(tensor))::value % 2 == 0);
    const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
    const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
    // if (cute::thread0()) { printf("threshold2 = 0x%x\n", p_dropout_8bit_in_uint32_t); }
    #pragma unroll
    for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
        uint2 rowcol = make_uint2(block_row_start, block_col_start);
        #pragma unroll
        for (int n = 0; n < size<2>(tensor) / 2; ++n, ++rowcol.y) {
            // if (cute::thread(32, 0)) { printf("m = %d, n = %d, row = %d, col = %d\n", m, n, int(rowcol.x), int(rowcol.y));}
            uint4 random_uint4 = pytorch_flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
            // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
            uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
            // Special implementation for 16-bit types: we duplicate the threshold to the
            // low and high 16 bits of a 32-bit value, then use the f16x2 comparison instruction
            // to get a mask. The low 16 bits of the mask will be either 0xffff or 0x0000,
            // and the high 16 bits will be either 0xffff or 0x0000, depending on whether
            // the random value is less than the threshold.
            // We then do a bit-wise AND between the mask and the original value (in 32-bit).
            // We're exploiting the fact that floating point comparison is equivalent to integer
            // comparison, since we're comparing unsigned integers whose top 8-bits are zero.
            if (!encode_dropout_in_sign_bit
                && (std::is_same<T, cutlass::half_t>::value || std::is_same<T, cutlass::bfloat16_t>::value)) {
                uint16_t rnd_16[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                uint32_t (&rnd_32)[8] = reinterpret_cast<uint32_t (&)[8]>(rnd_16);
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                    // if (cute::thread0()) { printf("random = 0x%x, 0x%x, 0x%x, 0x%x\n", rnd_32[j * 4 + 0], rnd_32[j * 4 + 1], rnd_32[j * 4 + 2], rnd_32[j * 4 + 3]); }
                    // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        uint32_t mask;
                        asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[j * 4 + i]), "r"(p_dropout_8bit_in_uint32_t));
                        tensor_uint32(i) &= mask;
                    }
                    // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        tensor(i, m, n * 2 + j) = encode_dropout(rnd_8[j * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n * 2 + j));
                    }
                    Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                    // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                }
            }
            // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
            // //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
            // // }
        }
    }
}

}  // namespace pytorch_flash
