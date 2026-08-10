#pragma once

#include <torch/headeronly/cuda/Atomic.h>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

#if defined(USE_ROCM)
#include <device_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <torch/headeronly/cuda/detail/ROCmMacros.h>

#define ATOMICADD unsafeAtomicAdd
#define NATIVE_ZERO_BF16 __float2bfloat16(0.0f)
#else
#define ATOMICADD atomicAdd
#define NATIVE_ZERO_BF16 __int2bfloat16_rz(0)
#endif

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// fastSpecializedAtomicAdd (and fastAtomicAdd) are an optimization
// that speed up half-precision atomics.  The situation with half
// precision atomics is that we have a slow __half atomic, and
// a fast vectored __half2 atomic (this can be worth up to a 6x
// speedup, see https://github.com/pytorch/pytorch/pull/21879).
// We can convert a __half atomic into a __half2 atomic by simply
// pairing the __half with a zero entry on the left/right depending
// on alignment... but only if this wouldn't cause an out of bounds
// access!  Thus, you must specify tensor and numel so we can check
// if you would be out-of-bounds and use a plain __half atomic if
// you would be.
template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if_t<std::is_same_v<c10::Half, scalar_t>>* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  gpuAtomicAddNoReturn(
      reinterpret_cast<at::Half*>(tensor) + index,
      static_cast<at::Half>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32
  // bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(tensor + index);
  bool low_byte =
      (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = static_cast<__half>(value);
    value2.y = __int2half_rz(0);
    ATOMICADD(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = static_cast<__half>(value);
    ATOMICADD(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
#ifdef USE_ROCM
    gpuAtomicAddNoReturn(
        reinterpret_cast<at::Half*>(tensor) + index,
        static_cast<at::Half>(value));
#else
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
#endif
  }
#endif
}

template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if_t<std::is_same_v<c10::BFloat16, scalar_t>>* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  gpuAtomicAddNoReturn(
      reinterpret_cast<at::BFloat16*>(tensor) + index,
      static_cast<at::BFloat16>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32
  // bit aligned)
  __nv_bfloat16* target_addr = reinterpret_cast<__nv_bfloat16*>(tensor + index);
  bool low_byte =
      (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__nv_bfloat162) ==
       0);

  if (low_byte && index < (numel - 1)) {
    __nv_bfloat162 value2;
    value2.x = *reinterpret_cast<__nv_bfloat16*>(&value);
    value2.y = NATIVE_ZERO_BF16;
    ATOMICADD(reinterpret_cast<__nv_bfloat162*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __nv_bfloat162 value2;
    value2.x = NATIVE_ZERO_BF16;
    value2.y = *reinterpret_cast<__nv_bfloat16*>(&value);
    ATOMICADD(reinterpret_cast<__nv_bfloat162*>(target_addr - 1), value2);

  } else {
#ifdef USE_ROCM
    gpuAtomicAddNoReturn(
        reinterpret_cast<at::BFloat16*>(tensor) + index,
        static_cast<at::BFloat16>(value));
#else
    atomicAdd(
        reinterpret_cast<__nv_bfloat16*>(tensor) + index,
        *reinterpret_cast<__nv_bfloat16*>(&value));
#endif
  }
#endif
}

template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if_t<
        !std::is_same_v<c10::Half, scalar_t> &&
        !std::is_same_v<c10::BFloat16, scalar_t>>* = nullptr>
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

HIDDEN_NAMESPACE_END(torch, headeronly)

// ATOMICADD/NATIVE_ZERO_BF16 are only expanded above; don't leak them.
#undef ATOMICADD
#undef NATIVE_ZERO_BF16
