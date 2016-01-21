/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/


#ifndef REDUCE_KERNEL_H_
#define REDUCE_KERNEL_H_

#include "common_kernel.h"
#include <limits>

template<typename T>
struct FuncSum {
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd {
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncMax {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin {
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};

template<>
struct FuncSum<char> {
  union converter {
    uint32_t storage;
    char4 a;
  };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    asm("vadd4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500)
    int32_t rv;
    asm("vadd.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
        "vadd.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
        "vadd.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
        "vadd.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;" : "=r"(rv) : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = cx.a.x + cy.a.x;
    cr.a.y = cx.a.y + cy.a.y;
    cr.a.z = cx.a.z + cy.a.z;
    cr.a.w = cx.a.w + cy.a.w;
    return cr.storage;
#endif
  }
  __device__ char operator()(const char x, const char y) const {
    return x+y;
  }
};

template<>
struct FuncProd<char> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300)
    int32_t rv, zero=0;
    asm("{ .reg .u32 t0, t1, t2, t3;\n\t"
        " vmad.u32.u32.u32 t3, %1.b3, %2.b3, %3;\n\t"
        " vmad.u32.u32.u32 t2, %1.b2, %2.b2, %3;\n\t"
        " shl.b32          t3, t3, 16;\n\t"
        " shl.b32          t2, t2, 16;\n\t"
        " vmad.u32.u32.u32 t1, %1.b1, %2.b1, t3;\n\t"
        " shl.b32          t1, t1, 8;\n\t"
        " vmad.u32.u32.u32 t0, %1.b0, %2.b0, t2;\n\t"
        " and.b32          t1, t1, 0xff00ff00;\n\t"
        " and.b32          t0, t0, 0x00ff00ff;\n\t"
        " or.b32           %0,  t0, t1;\n\t"
        "}" : "=r"(rv) : "r"(x), "r"(y), "r"(zero));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = cx.a.x * cy.a.x;
    cr.a.y = cx.a.y * cy.a.y;
    cr.a.z = cx.a.z * cy.a.z;
    cr.a.w = cx.a.w * cy.a.w;
    return cr.storage;
#endif
  }
  __device__ char operator()(const char x, const char y) const {
    return x*y;
  }
};

template<>
struct FuncMax<char> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    if (std::numeric_limits<char>::is_signed)
      asm("vmax4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    else
      asm("vmax4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500)
    int32_t rv;
    if (std::numeric_limits<char>::is_signed)
      asm("vmax.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
          "vmax.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
          "vmax.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
          "vmax.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;" : "=r"(rv) : "r"(x), "r"(y));
    else
      asm("vmax.u32.u32.u32 %0,    %1.b0, %2.b0;    \n\t"
          "vmax.u32.u32.u32 %0.b1, %1.b1, %2.b1, %0;\n\t"
          "vmax.u32.u32.u32 %0.b2, %1.b2, %2.b2, %0;\n\t"
          "vmax.u32.u32.u32 %0.b3, %1.b3, %2.b3, %0;" : "=r"(rv) : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = max(cx.a.x, cy.a.x);
    cr.a.y = max(cx.a.y, cy.a.y);
    cr.a.z = max(cx.a.z, cy.a.z);
    cr.a.w = max(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ char operator()(const char x, const char y) const {
    return (x>y) ? x : y;
  }
};

template<>
struct FuncMin<char> {
  union converter { uint32_t storage; char4 a; };
  __device__ uint32_t operator()(const uint32_t x, const uint32_t y) const {
#if (__CUDA_ARCH__ >= 300) && (__CUDA_ARCH__ < 500)
    int32_t rv, z=0;
    if (std::numeric_limits<char>::is_signed)
      asm("vmin4.s32.s32.s32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    else
      asm("vmin4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(x), "r"(y), "r"(z));
    return rv;
#elif (__CUDA_ARCH__ >= 500)
    int32_t rv;
    if (std::numeric_limits<char>::is_signed)
      asm("vmin.s32.s32.s32 %0,    %1.b0, %2.b0;    \n\t"
          "vmin.s32.s32.s32 %0.b1, %1.b1, %2.b1, %0;\n\t"
          "vmin.s32.s32.s32 %0.b2, %1.b2, %2.b2, %0;\n\t"
          "vmin.s32.s32.s32 %0.b3, %1.b3, %2.b3, %0;" : "=r"(rv) : "r"(x), "r"(y));
    else
      asm("vmin.u32.u32.u32 %0,    %1.b0, %2.b0;    \n\t"
          "vmin.u32.u32.u32 %0.b1, %1.b1, %2.b1, %0;\n\t"
          "vmin.u32.u32.u32 %0.b2, %1.b2, %2.b2, %0;\n\t"
          "vmin.u32.u32.u32 %0.b3, %1.b3, %2.b3, %0;" : "=r"(rv) : "r"(x), "r"(y));
    return rv;
#else
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a.x = min(cx.a.x, cy.a.x);
    cr.a.y = min(cx.a.y, cy.a.y);
    cr.a.z = min(cx.a.z, cy.a.z);
    cr.a.w = min(cx.a.w, cy.a.w);
    return cr.storage;
#endif
  }
  __device__ char operator()(const char x, const char y) const {
    return (x<y) ? x : y;
  }
};

#ifdef CUDA_HAS_HALF
template<>
struct FuncSum<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x + fy.x;
    fr.y = fx.y + fy.y;
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    return __float2half( __half2float(x) + __half2float(y) );
  }
};

template<>
struct FuncProd<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x * fy.x;
    fr.y = fx.y * fy.y;
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    return __float2half( __half2float(x) * __half2float(y) );
  }
};

template<>
struct FuncMax<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x > fy.x ? fx.x : fy.x;
    fr.y = fx.y > fy.y ? fx.y : fy.y;
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fx > fy ? fx : fy;
    return __float2half(fm);
  }
};

template<>
struct FuncMin<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    float2 fx, fy, fr;
    fx = __half22float2(x);
    fy = __half22float2(y);
    fr.x = fx.x < fy.x ? fx.x : fy.x;
    fr.y = fx.y < fy.y ? fx.y : fy.y;
    return __float22half2_rn(fr);
  }
  __device__ half operator()(const half x, const half y) const {
    float fx, fy, fm;
    fx = __half2float(x);
    fy = __half2float(y);
    fm = fx < fy ? fx : fy;
    return __float2half(fm);
  }
};
#endif

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of threads in the CTA
// - this function is called by all producer threads
template<int UNROLL, int THREADS, class FUNC, typename T>
__device__ void Reduce(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  ReduceOrCopy<UNROLL, THREADS, FUNC, T, false, true>(threadIdx.x, dest,
      nullptr, src0, src1, N);
}

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of threads in the CTA
// - this function is called by all producer threads
template<int UNROLL, int THREADS, class FUNC, typename T>
__device__ void ReduceAndCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src0,
    const volatile T * __restrict__ const src1, const int N) {
  ReduceOrCopy<UNROLL, THREADS, FUNC, T, true, true>(threadIdx.x, dest0, dest1,
      src0, src1, N);
}

#endif // REDUCE_KERNEL_H_
