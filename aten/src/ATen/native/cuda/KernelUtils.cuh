#pragma once
#include <ATen/cuda/Atomic.cuh>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

namespace at {
namespace native {

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t h,
    const size_t w) {
  return (nc * height + h) * width + w;
}

// for channels-last
__device__ __forceinline__ size_t
idx_cl(
  const size_t n, const size_t h, const size_t w, const size_t c,
  const size_t height, const size_t width, const size_t channel
) {
  return ((n * height + h) * width + w) * channel + c;
}

template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if<std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (                      \
    (defined(USE_ROCM)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  gpuAtomicAddNoReturn(
      reinterpret_cast<at::Half*>(tensor) + index,
      static_cast<at::Half>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(tensor + index);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if<std::is_same<c10::BFloat16, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (                      \
    (defined(USE_ROCM)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)))
  gpuAtomicAddNoReturn(
      reinterpret_cast<at::BFloat16*>(tensor) + index,
      static_cast<at::BFloat16>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __nv_bfloat16* target_addr = reinterpret_cast<__nv_bfloat16*>(tensor + index);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__nv_bfloat162) == 0);

  if (low_byte && index < (numel - 1)) {
    __nv_bfloat162 value2;
    value2.x = *reinterpret_cast<__nv_bfloat16*>(&value);
    value2.y = __int2bfloat16_rz(0);
    atomicAdd(reinterpret_cast<__nv_bfloat162*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __nv_bfloat162 value2;
    value2.x = __int2bfloat16_rz(0);
    value2.y = *reinterpret_cast<__nv_bfloat16*>(&value);
    atomicAdd(reinterpret_cast<__nv_bfloat162*>(target_addr - 1), value2);

  } else {
    atomicAdd(
        reinterpret_cast<__nv_bfloat16*>(tensor) + index, *reinterpret_cast<__nv_bfloat16*>(&value));
  }
#endif
}


template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if<!std::is_same<c10::Half, scalar_t>::value && !std::is_same<c10::BFloat16, scalar_t>::value >::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
  gpuAtomicAddNoReturn(tensor + index, value);
}

template <class scalar_t, class index_t>
__device__ __forceinline__ void fastAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value,
    bool fast_atomics) {
  if (fast_atomics) {
    fastSpecializedAtomicAdd(tensor, index, numel, value);
  } else {
    gpuAtomicAddNoReturn(tensor + index, value);
  }
}

} // namespace native
} // namespace at
