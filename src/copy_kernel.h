/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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


#ifndef COPY_KERNEL_H_
#define COPY_KERNEL_H_

#include "common_kernel.h"

template<typename T>
struct FuncPassA {
  __device__ T operator()(const T x, const T y) const {
    return x;
  }
};

#ifdef CUDA_HAS_HALF
template <>
struct FuncPassA<half> {
  __device__ half2 operator()(const half2 x, const half2 y) const {
    return x;
  }
  __device__ half operator()(const half x, const half y) const {
    half r;
    r.x = x.x;
    return r;
  }
};
#endif

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of producer threads
// - this function is called by all producer threads
template<int UNROLL, int THREADS, typename T>
__device__ void Copy(volatile T * __restrict__ const dest,
    const volatile T * __restrict__ const src, const int N) {
  ReduceOrCopy<UNROLL, THREADS, FuncPassA<T>, T, false, false>(threadIdx.x,
      dest, nullptr, src, nullptr, N);
}

// Assumptions:
// - there is exactly 1 block
// - THREADS is the number of producer threads
// - this function is called by all producer threads
template<int UNROLL, int THREADS, typename T>
__device__ void DoubleCopy(volatile T * __restrict__ const dest0,
    volatile T * __restrict__ const dest1,
    const volatile T * __restrict__ const src, const int N) {
  ReduceOrCopy<UNROLL, THREADS, FuncPassA<T>, T, true, false>(threadIdx.x,
      dest0, dest1, src, nullptr, N);
}

#endif // COPY_KERNEL_H_
