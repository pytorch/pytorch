#pragma once
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include <ATen/ATen.h>
//#include <ATen/AccumulateType.h>
//#include <ATen/NativeFunctions.h>
//#include <ATen/TensorUtils.h>
//#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAApplyUtils.cuh>
//#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {

template <
    typename scalar_t,
    typename std::enable_if<std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value) {
#if (                         \
    (CUDA_VERSION < 10000) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(
      reinterpret_cast<at::Half*>(tensor) + index,
      static_cast<at::Half>(value));
#else
  if (index % 2 == 0 && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);

  } else if (index % 2 == 1) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(tensor) + index / 2, value2);

  } else {
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

template <
    typename scalar_t,
    typename std::enable_if<!std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value) {
  atomicAdd(tensor + index, value);
}

template <class scalar_t>
__device__ __forceinline__ void fastAtomicAdd(
    scalar_t* tensor,
    size_t index,
    const size_t numel,
    scalar_t value,
    bool fast_atomics) {
  if (fast_atomics) {
    fastSpecializedAtomicAdd(tensor, index, numel, value);
  } else {
    atomicAdd(tensor + index, value);
  }
}

} // namespace native
} // namespace at
