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

#include <cutlass/cutlass.h>

#include <cutlass/gemm/gemm.h>

#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include <ATen/native/transformers/cuda/flash_attn/gemm.h>
#include <ATen/native/transformers/cuda/flash_attn/gmem_tile.h>
#include <ATen/native/transformers/cuda/flash_attn/summary_stats.h>
#include <ATen/native/transformers/cuda/flash_attn/mma_core_sm75.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int S, int D, int STEP, int WARPS_M, int WARPS_N, uint32_t FLAGS = 0x08u, typename elem_type=cutlass::half_t>
struct FMHA_kernel_traits {

    // The CTA description for the 1st GEMM.
    using Cta_tile_p = fmha::Cta_tile_extd<STEP, S, D, WARPS_M, WARPS_N, 1>;
    // The CTA description for the 2nd GEMM.
    using Cta_tile_o = fmha::Cta_tile_extd<STEP, D, S, WARPS_M, 1, WARPS_N>;

    // Do we use one buffer for K and V.
    static constexpr bool SHARE_SMEM_FOR_K_AND_V = (FLAGS & 0x08u) != 0u;
    // Do we keep K in registers.
    static constexpr bool K_IN_REGS = (FLAGS & 0x10u) == 0u;
    // Do we keep V in registers.
    static constexpr bool V_IN_REGS = (FLAGS & 0x100u) == 0u;

    // The global memory tile to load/store S.
    using Gmem_tile_s = fmha::Gmem_tile_mma_s<Cta_tile_p>;

    // The global memory tile to store the softmax sum.
    using Gmem_softmax_sum = fmha::Gmem_summary_stats<Cta_tile_p>;

    // The number of threads.
    static constexpr int THREADS = Cta_tile_p::THREADS_PER_CTA;
    // Make sure the number of threads matches both CTAs.
    static_assert(THREADS == Cta_tile_o::THREADS_PER_CTA, "");

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using MmaInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using MmaInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
#else
    // using MmaInstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
    using MmaInstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    // TD [2022-06-02] We don't support Volta (SM70) yet.
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
#else
    using Element = cutlass::half_t;
#endif
    using ElementAccum = float;

    static_assert(WARPS_M == 1, "");
    using ThreadblockShapeQK = cutlass::gemm::GemmShape<STEP, S, D>;
    using WarpCountQK = cutlass::gemm::GemmShape<WARPS_M, WARPS_N, 1>;
    using WarpShapeQK = cutlass::gemm::GemmShape<
       ThreadblockShapeQK::kM,
       ThreadblockShapeQK::kN / WarpCountQK::kN, ThreadblockShapeQK::kK>;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutP = cutlass::layout::RowMajor;
    using MmaCoreQK = typename fmha::FMHAMmaCore<
        ThreadblockShapeQK, WarpShapeQK, MmaInstructionShape, Element, LayoutQ,
        Element, LayoutK, ElementAccum, LayoutP,
        cutlass::arch::OpClassTensorOp>;

    using ThreadblockShapePV = cutlass::gemm::GemmShape<STEP, D, S>;
    using WarpCountPV = cutlass::gemm::GemmShape<WARPS_M, 1, WARPS_N>;
    using WarpShapePV = cutlass::gemm::GemmShape<ThreadblockShapePV::kM, ThreadblockShapePV::kN, ThreadblockShapePV::kK / WarpCountPV::kK>;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;
    using MmaCorePV = typename fmha::FMHAMmaCore<
        ThreadblockShapePV, WarpShapePV, MmaInstructionShape, Element, LayoutP,
        Element, LayoutV, ElementAccum, LayoutO,
        cutlass::arch::OpClassTensorOp>;

    // The global memory tile to load Q.
    // Copy from mma_piplined_testbed.h
    using GmemIteratorQ = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapeQK::kM, ThreadblockShapeQK::kK>,
      Element,
      LayoutQ,
      0,
      typename MmaCoreQK::IteratorThreadMapA
    >;

    // The global memory tile to load K.
    using GmemIteratorK = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapeQK::kK, ThreadblockShapeQK::kN>,
      Element,
      LayoutK,
      1,
      typename MmaCoreQK::IteratorThreadMapB
    >;

    // The global memory tile to load V.
    using GmemIteratorV = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadblockShapePV::kK, ThreadblockShapePV::kN>,
      Element,
      LayoutV,
      0,
      typename MmaCorePV::IteratorThreadMapB
    >;

    // The shared memory tile to store softmax lse.
    using Smem_softmax_lse = fmha::Smem_tile_softmax_lse<ThreadblockShapeQK::kM, MmaInstructionShape::kM, WarpCountQK::kM>;

    // The amount of shared memory needed to load Q and K.
    static constexpr size_t BYTES_PER_SMEM_Q = ThreadblockShapeQK::kM * ThreadblockShapeQK::kK * sizeof(Element);
    static constexpr size_t BYTES_PER_SMEM_K = ThreadblockShapeQK::kN * ThreadblockShapeQK::kK * sizeof(Element);
    static constexpr size_t BYTES_PER_SMEM_V = ThreadblockShapePV::kN * ThreadblockShapePV::kK * sizeof(Element);
    static_assert(BYTES_PER_SMEM_K == BYTES_PER_SMEM_V, "");
    static constexpr size_t BYTES_PER_SMEM_QK = BYTES_PER_SMEM_Q + BYTES_PER_SMEM_K;
    // The extra amount of shared memory needed to load V.
    static constexpr size_t BYTES_PER_SMEM_V_EXTRA = SHARE_SMEM_FOR_K_AND_V ? 0u : BYTES_PER_SMEM_V;
    // The amount of shared memory needed for Q, K and V..
    static constexpr size_t BYTES_PER_SMEM_QKV = BYTES_PER_SMEM_QK + BYTES_PER_SMEM_V_EXTRA;

};
