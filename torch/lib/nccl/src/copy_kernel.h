/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
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
    return x;
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
