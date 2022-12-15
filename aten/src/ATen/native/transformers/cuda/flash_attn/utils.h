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

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
// #include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

extern "C" __device__ uint32_t __nvvm_get_smem_pointer(void *ptr);

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Row {};
struct Col {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int M, int N >
struct Div_up {
    enum { VALUE = (M + N-1) / N };
};

constexpr int DivUpConstexpr(int M, int N) { return (M + N - 1) / N; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B >
struct Max {
    enum { VALUE = A >= B ? A : B };
};

constexpr int MaxConstexpr(int A, int B) { return A >= B ? A : B; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B, int C >
struct Max_3 {
    enum { VALUE = Max<Max<A, B>::VALUE, C>::VALUE };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B >
struct Min {
    enum { VALUE = A <= B ? A : B };
};

constexpr int MinConstexpr(int A, int B) { return A <= B ? A : B; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES >
struct Uint_from_size_in_bytes {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<1> {
    using Type = uint8_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<2> {
    using Type = uint16_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<4> {
    using Type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<8> {
    using Type = uint2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Uint_from_size_in_bytes<16> {
    using Type = uint4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
inline __device__ __host__ T div_up(T m, T n) {
    return (m + n-1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ uint32_t hrelu2(uint32_t x);

template<>
inline __device__ uint32_t hrelu2<__half>(uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile( "max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n"
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
inline __device__ uint32_t hrelu2<__nv_bfloat16>(uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile( "max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint16_t float_to_half(float f) {
    uint16_t h;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ uint32_t float2_pack(float a, float b);

template <>
inline __device__ uint32_t float2_pack<__half>(float a, float b) {
    __half2 result = __floats2half2_rn(a, b);
    return reinterpret_cast<uint32_t(&)>(result);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
inline __device__ uint32_t float2_pack<__nv_bfloat16>(float a, float b) {
    __nv_bfloat162 result = __floats2bfloat162_rn(a, b);
    return reinterpret_cast<uint32_t(&)>(result);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ uint2 float4_pack(float x, float y, float z, float w) {
    uint2 d;
    d.x = float2_pack<T>(x, y);
    d.y = float2_pack<T>(z, w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ float2 half2_unpack(uint32_t a);

template <>
inline __device__ float2 half2_unpack<__half>(uint32_t a) {
    return __half22float2(reinterpret_cast<__half2 (&)>(a));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
inline __device__ float2 half2_unpack<__nv_bfloat16>(uint32_t a) {
    return __bfloat1622float2(reinterpret_cast<__nv_bfloat162 (&)>(a));
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert two half2's or bf162's into float, then take their dot product.
template <typename T>
inline __device__ float hfma2_to_float(const uint32_t a, const uint32_t b) {
    float2 af = fmha::half2_unpack<T>(a);
    float2 bf = fmha::half2_unpack<T>(b);
    return af.x * bf.x + af.y * bf.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Converted two vectors of 8 half's or bf16's into float, then take their dot product.
template<typename T>
inline __device__ float hmulsum8(const uint4 a, const uint4 b) {
    float sum;
    sum  = fmha::hfma2_to_float<T>(a.x, b.x);
    sum += fmha::hfma2_to_float<T>(a.y, b.y);
    sum += fmha::hfma2_to_float<T>(a.z, b.z);
    sum += fmha::hfma2_to_float<T>(a.w, b.w);
    return sum;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// L D G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint8_t &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint8_t*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint16_t &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint16_t*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint32_t &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint32_t*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint2 &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint2*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void ldg(uint4 &dst, const void *ptr) {
    dst = *reinterpret_cast<const uint4*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S T G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint8_t val) {
    *reinterpret_cast<uint8_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint16_t val) {
    *reinterpret_cast<uint16_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint32_t val) {
    *reinterpret_cast<uint32_t*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint2 val) {
    *reinterpret_cast<uint2*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void stg(void *ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4, "");
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Operator, int M>
__device__ inline void  quad_reduce(float (&dst)[M], float (&src)[M], Operator &op) {
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        dst[mi] = src[mi];
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 2));
        dst[mi] = op(dst[mi], __shfl_down_sync(uint32_t(-1), dst[mi], 1));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Operator, int M>
__device__ inline void quad_reduce(float (&dst)[M], float2 (&src)[M], Operator &op) {
    float tmp[M];
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        tmp[mi] = op(src[mi].x, src[mi].y);
    }
    quad_reduce(dst, tmp, op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Operator, int M>
__device__ inline void quad_allreduce(float (&dst)[M], float (&src)[M], Operator &op) {
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        dst[mi] = src[mi];
        dst[mi] = Allreduce<4>::run(dst[mi], op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Operator, int M>
__device__ inline void quad_allreduce(float (&dst)[M], float2 (&src)[M], Operator &op) {
    float tmp[M];
    #pragma unroll
    for(int mi=0; mi < M; mi++){
        tmp[mi] = op(src[mi].x, src[mi].y);
    }
    quad_allreduce(dst, tmp, op);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace fmha
