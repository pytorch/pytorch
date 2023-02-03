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

#include <ATen/native/transformers/cuda/flash_attn/utils.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/layout/layout.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type_, int NUM_ELTS_, int BITS_PER_ELT_, int ALIGNMENT_ >
struct Fragment_base_ {

    // The data type.
    using Data_type = Data_type_;
    // default input type
    using Input_type_ = Data_type_;
    // Does it store the array of elements.
    static constexpr bool HAS_ELTS = BITS_PER_ELT_ >= 8;
    // The number of elements.
    static constexpr int NUM_ELTS = NUM_ELTS_;
    // The size of element in bits.
    static constexpr int BITS_PER_ELT = BITS_PER_ELT_;
    // The size of byte of a single register.
    static constexpr int BYTES_PER_REG = 4;
    // The size in bits.
    static constexpr int BITS_PER_REG = BYTES_PER_REG * 8;
    // The number of registers needed to store the fragment.
    static constexpr int NUM_REGS = DivUpConstexpr(NUM_ELTS * BITS_PER_ELT, BITS_PER_REG);
    // The size in bytes (as returned by sizeof(Fragment_base<>).
    static constexpr int SIZE_IN_BYTES = NUM_REGS * BYTES_PER_REG;
    // The alignment.
    static constexpr int ALIGNMENT = ALIGNMENT_ > 0 ? ALIGNMENT_ : MinConstexpr(NUM_REGS * BYTES_PER_REG, 16);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The type of the elements.
    typename Data_type_,
    // The number of elements.
    int NUM_ELTS_,
    // The alignment if you want to force a value -- use 0 otherwise.
    int ALIGNMENT_ = 0,
    // The base class.
    typename Base_ = Fragment_base_<Data_type_, NUM_ELTS_, 8 * sizeof(Data_type_), ALIGNMENT_>
>
struct alignas(static_cast<int>(Base_::ALIGNMENT)) Fragment : public Base_ {

    // The size of a load/store.
    static constexpr int BYTES_PER_LOAD_STORE = Base_::NUM_REGS * sizeof(uint32_t);

    // Clear the fragment. Using PTX in that code seems to produce better SASS...
    inline __device__ void clear() {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
            asm volatile("mov.u32 %0, 0; \n" : "=r"(this->reg(ii)) : );
        }
    }

    // Immutable access to a register.
    inline __device__ const uint32_t& reg(int ii) const {
        return this->regs_[ii];
    }

    // Mutable access to a register.
    inline __device__ uint32_t& reg(int ii) {
        return this->regs_[ii];
    }

    uint32_t regs_[Base_::NUM_REGS];

    // Immutable access to the elements.
    inline __device__ const Data_type_& elt(int ii) const {
        return reinterpret_cast<const Data_type_*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    inline __device__ Data_type_& elt(int ii) {
        return reinterpret_cast<Data_type_*>(&this->regs_[0])[ii];
    }

    // Immutable access to the elements with a cast.
    template< typename Cast_type >
    inline __device__ const Cast_type& elt_as(int ii) const {
        return reinterpret_cast<const Cast_type*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    template< typename Cast_type >
    inline __device__ Cast_type& elt_as(int ii) {
        return reinterpret_cast<Cast_type*>(&this->regs_[0])[ii];
    }

    // Add another fragment.
    inline __device__ void add(const Fragment &other) {
        // TODO (TD 2022-04-09): Shouldn't this be NUM_REGS instead of NUM_ELTS?
        // Also are we doing int addition or __half2 addition?
        #pragma unroll
        for( int ii = 0; ii < NUM_ELTS_; ++ii ) {
            this->elt(ii) += other.elt(ii);
        }
    }

    // Multiply by another fragment.
    inline __device__ void hmul(const Fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
            this->reg(ii) = fmha::hmul2(this->reg(ii), other.reg(ii));
        }
    }

    template <typename elem_type>
    inline __device__ void hrelu_() {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
            this->reg(ii) = fmha::hrelu2<elem_type>(this->reg(ii));
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_a : public Fragment<uint16_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Layout >
struct Fragment_b : public Fragment<uint16_t, 8> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fragment_accumulator : public Fragment<float, 8> {

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments.
    template< typename Other_fragment_ >
    inline __device__ void add(const Other_fragment_ &other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    inline __device__ void mul_(const float other) {
        for( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) *= other;
        }
    }

    // Do the HMMA.
    template< typename Layout_a, typename Layout_b >
    inline __device__ void mma(const Fragment_a<Layout_a> &a,
                               const Fragment_b<Layout_b> &b) {
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(0)), "+f"(  elt(1)), "+f"(  elt(2)), "+f"(  elt(3))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(0)),  "r"(b.reg(1)));
        asm volatile( \
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n" \
            "    {%0, %1, %2, %3}, \n" \
            "    {%4, %5, %6, %7}, \n" \
            "    {%8, %9}, \n" \
            "    {%0, %1, %2, %3}; \n" \
                    : "+f"(  elt(4)), "+f"(  elt(5)), "+f"(  elt(6)), "+f"(  elt(7))
                    :  "r"(a.reg(0)),  "r"(a.reg(1)),  "r"(a.reg(2)),  "r"(a.reg(3))
                    ,  "r"(b.reg(2)),  "r"(b.reg(3)));
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Fragment, int M, int N >
inline __device__ void clear(Fragment (&frag)[M][N]) {
    #pragma unroll
    for( int mi = 0; mi < M; ++mi ) {
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            frag[mi][ni].clear();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Accumulator_type, int WARPS_K >
struct Clear_accumulator {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int WARPS_K >
struct Clear_accumulator<float, WARPS_K> {
  template< typename Acc, int M, int N >
  static inline __device__ void apply(Acc (&acc)[M][N], bool = false) {
    fmha::clear(acc);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Acc, typename A, typename B, int M, int N>
inline __device__ void gemm(Acc (&acc)[M][N], const A (&a)[M], const B (&b)[N]) {

    #pragma unroll
    for( int mi = 0; mi < M; ++mi ) {
        #pragma unroll
        for( int ni = 0; ni < N; ++ni ) {
            acc[mi][ni].mma(a[mi], b[ni]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Statically maps half types => cutlass data types
/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Type_>
struct HalfTypeToCutlassType { using Type = Type_; };

/// Statically maps __half => cutlass::half_t
template <> struct HalfTypeToCutlassType<__half> {
    using Type = cutlass::half_t;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <> struct HalfTypeToCutlassType<__nv_bfloat16> {
    using Type = cutlass::bfloat16_t;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename elem_type, typename Acc, typename A, typename B, int M, int N>
inline __device__ void gemm_cl(Acc (&acc)[M][N], const A (&a)[M], const B (&b)[N]) {
    using Shape = cutlass::gemm::GemmShape<16 * M, 16 * N, 16>;
#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
#elif defined(__CUDA_ARCH__)  && __CUDA_ARCH__ >= 750
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
#else
    assert(0);
    // THIS IS NOT CORRECT BUT THE ASSERT WILL STOP THIS
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    // TD [2022-06-02] We don't support Volta (SM70) yet.
#endif
    using Element = typename HalfTypeToCutlassType<elem_type>::Type;
    using ElementC = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;

    using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
        cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd, 1, true>::Type;

    constexpr int kIters = Shape::kK / InstructionShape::kK;
    // using FragmentA = typename WarpMma::FragmentA;
    // using FragmentB = typename WarpMma::FragmentB;
    using FragmentA = typename WarpMma::ArchMmaOperator::FragmentA;
    using FragmentB = typename WarpMma::ArchMmaOperator::FragmentB;
    using FragmentC = typename WarpMma::FragmentC;

    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y) == 0) {
    //     printf("FragmentA::kStorageElements = %d\n", FragmentA::kStorageElements);
    //     printf("Archmma::FragmentA::kStorageElements = %d\n", WarpMma::ArchMmaOperator::FragmentA::kStorageElements);
    //     printf("FragmentB::kStorageElements = %d\n", FragmentB::kStorageElements);
    //     printf("Archmma::FragmentB::kStorageElements = %d\n", WarpMma::ArchMmaOperator::FragmentB::kStorageElements);
    //     printf("FragmentC::kStorageElements = %d\n", FragmentC::kStorageElements);
    //     printf("Archmma::FragmentC::kStorageElements = %d\n", WarpMma::ArchMmaOperator::FragmentC::kStorageElements);
    // }

    // static_assert(FragmentA::kStorageElements == M * a[0].NUM_REGS);
    // static_assert(FragmentB::kStorageElements == N * b[0].NUM_REGS);
    static_assert(FragmentA::kStorageElements * kIters == a[0].NUM_REGS);
    static_assert(FragmentB::kStorageElements * kIters * 16 / InstructionShape::kN == b[0].NUM_REGS);
    static_assert(FragmentC::kStorageElements == M * N * acc[0][0].NUM_REGS);
    // const FragmentA a_cl = reinterpret_cast<const FragmentA (&)>(a);
    // const FragmentB b_cl = reinterpret_cast<const FragmentB (&)>(b);
    FragmentC c_cl = reinterpret_cast<FragmentC (&)>(acc);
    FragmentA a_cl[kIters][M];
    FragmentA b_cl[kIters][N];
    constexpr int kRegs = InstructionShape::kK == 16 ? 4 : 2;
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        #pragma unroll
        for (int mi = 0; mi < M; mi++) {
            uint32_t *a_ptr = a_cl[iter][mi].raw_data();
            #pragma unroll
            for (int ki = 0; ki < kRegs; ki++) {
                a_ptr[ki] = a[mi].regs_[iter * kRegs + ki];
            }
        }
    }
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            uint32_t *b_ptr = b_cl[iter][ni].raw_data();
            #pragma unroll
            for (int ki = 0; ki < kRegs; ki++) {
                // b_ptr[ki] = b[ni].regs_[iter * kRegs + ki];
                // TD [2022-06-02] For some reason the order for frag_b is different.
                b_ptr[ki] = b[ni].regs_[InstructionShape::kK == 16 ? iter * kRegs + ki : ki * kRegs + iter];
            }
        }
    }

    WarpMma mma_op;
    // mma_op(c_cl, a_cl, b_cl, c_cl);
    #pragma unroll
    for (int iter = 0; iter < kIters; iter++) {
        mma_op(c_cl, reinterpret_cast<const typename WarpMma::FragmentA (&)>(a_cl[iter]),
               reinterpret_cast<const typename WarpMma::FragmentB (&)>(b_cl[iter]), c_cl);
    }

    // The modified c_cl is not copied back into acc, idk why
    #pragma unroll
    for (int mi = 0; mi < M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N; ni++) {
            #pragma unroll
            for (int i =0; i < 8; i++) {
                acc[mi][ni].elt(i) = c_cl[mi * N * 8 + ni * 8 + i];
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The number of rows in the CTA tile.
    int M_,
    // The number of cols in the CTA tile.
    int N_,
    // The number of elements in the the K dimension of the GEMM loop.
    int K_,
    // The number of rows of warps.
    int WARPS_M_,
    // The number of cols of warps.
    int WARPS_N_,
    // The number of warps in the K dimension of the GEMM loop.
    int WARPS_K_>
struct Cta_tile_ {

    static constexpr int M = M_, N = N_, K = K_;
    // The number of warps.
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_, WARPS_K = WARPS_K_;
    // The number of warps per CTA.
    static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N * WARPS_K;
    // The number of threads per warp.
    static constexpr int THREADS_PER_WARP = 32;
    // The number of threads per CTA.
    static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * THREADS_PER_WARP;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile>
struct Hmma_tile {
    // The number of elements computed with a single warp-MMA.
    static constexpr int M_PER_MMA = 16, N_PER_MMA = 16, K_PER_MMA = 16;

    // The number of elements computed with a single CTA-MMA.
    static constexpr int M_PER_MMA_PER_CTA = M_PER_MMA * Cta_tile::WARPS_M,
        N_PER_MMA_PER_CTA = N_PER_MMA * Cta_tile::WARPS_N,
        K_PER_MMA_PER_CTA = K_PER_MMA * Cta_tile::WARPS_K;

    // The number of MMAs needed to compute the GEMM.
    static constexpr int MMAS_M = DivUpConstexpr(Cta_tile::M, M_PER_MMA_PER_CTA),
        MMAS_N = DivUpConstexpr(Cta_tile::N, N_PER_MMA_PER_CTA),
        MMAS_K = DivUpConstexpr(Cta_tile::K, K_PER_MMA_PER_CTA);

    // // The number of elements computed per warp.
    // static constexpr int M_PER_WARP = MMAS_M * M_PER_MMA,
    //     N_PER_WARP = MMAS_N * N_PER_MMA,
    //     K_PER_WARP = MMAS_K * K_PER_MMA;

};

////////////////////////////////////////////////////////////////////////////////////////////////////

using A_type = uint16_t;
using B_type = uint16_t;
using C_type = uint16_t;
using Accumulator_type = float;
using Epilogue_type = float;

constexpr int BITS_PER_ELEMENT_A = sizeof(A_type) * 8;
constexpr int BITS_PER_ELEMENT_B = sizeof(B_type) * 8;
constexpr int BITS_PER_ELEMENT_C = sizeof(C_type) * 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int M, int N, int K, int WARPS_M, int WARPS_N, int WARPS_K>
using Cta_tile_extd = Cta_tile_<M, N, K, WARPS_M, WARPS_N, WARPS_K>;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Cta_tile_>
using Cta_tile_with_k_with_padding = Cta_tile_extd<Cta_tile_::M,
                                                   Cta_tile_::N,
                                                   Next_power_of_two<Cta_tile_::K>::VALUE,
                                                   Cta_tile_::WARPS_M,
                                                   Cta_tile_::WARPS_N,
                                                   Cta_tile_::WARPS_K>;

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
