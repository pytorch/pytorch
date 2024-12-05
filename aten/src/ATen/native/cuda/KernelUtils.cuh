#pragma once
#include <ATen/cuda/Atomic.cuh>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

// ROCm 6.3 is planned to have these functions, but until then here they are.
#if defined(USE_ROCM) && ROCM_VERSION >= 60201
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

__device__ inline __hip_bfloat162 preview_unsafeAtomicAdd(__hip_bfloat162* address, __hip_bfloat162 value) {
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) && \
  __has_builtin(__builtin_amdgcn_flat_atomic_fadd_v2bf16)
  typedef unsigned short __attribute__((ext_vector_type(2))) vec_short2;
  static_assert(sizeof(vec_short2) == sizeof(__hip_bfloat162_raw));
  union {
    __hip_bfloat162_raw bf162_raw;
    vec_short2 vs2;
  } u{static_cast<__hip_bfloat162_raw>(value)};
  u.vs2 = __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)address, u.vs2);
  return static_cast<__hip_bfloat162>(u.bf162_raw);
#else
  static_assert(sizeof(unsigned int) == sizeof(__hip_bfloat162_raw));
  union u_hold {
    __hip_bfloat162_raw h2r;
    unsigned int u32;
  };
  u_hold old_val, new_val;
  old_val.u32 = __hip_atomic_load((unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  do {
    new_val.h2r = __hadd2(old_val.h2r, value);
  } while (!__hip_atomic_compare_exchange_strong(
        (unsigned int*)address, &old_val.u32, new_val.u32,
        __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  return old_val.h2r;
#endif
}

__device__ inline __half2 preview_unsafeAtomicAdd(__half2* address, __half2 value) {
#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) && \
  __has_builtin(__builtin_amdgcn_flat_atomic_fadd_v2f16)
  // The api expects an ext_vector_type of half
  typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
  static_assert(sizeof(vec_fp162) == sizeof(__half2_raw));
  union {
    __half2_raw h2r;
    vec_fp162 fp16;
  } u {static_cast<__half2_raw>(value)};
  u.fp16 = __builtin_amdgcn_flat_atomic_fadd_v2f16((vec_fp162*)address, u.fp16);
  return static_cast<__half2>(u.h2r);
#else
  static_assert(sizeof(__half2_raw) == sizeof(unsigned int));
  union u_hold {
    __half2_raw h2r;
    unsigned int u32;
  };
  u_hold old_val, new_val;
  old_val.u32 = __hip_atomic_load((unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  do {
    new_val.h2r = __hadd2(old_val.h2r, value);
  } while (!__hip_atomic_compare_exchange_strong(
        (unsigned int*)address, &old_val.u32, new_val.u32,
        __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  return old_val.h2r;
#endif
}
#define ATOMICADD preview_unsafeAtomicAdd
#define NATIVE_ZERO_BF16 __float2bfloat16(0.0f)
#else
#define ATOMICADD atomicAdd
#define NATIVE_ZERO_BF16 __int2bfloat16_rz(0)
#endif

namespace at:: native {

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
    typename std::enable_if_t<std::is_same_v<c10::Half, scalar_t>>* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (                      \
    (defined(USE_ROCM) && ROCM_VERSION < 60201) || \
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
        reinterpret_cast<at::Half*>(tensor) + index, static_cast<at::Half>(value));
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
#if (                      \
    (defined(USE_ROCM) && ROCM_VERSION < 60201) || \
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
        reinterpret_cast<at::BFloat16*>(tensor) + index, static_cast<at::BFloat16>(value));
#else
    atomicAdd(
        reinterpret_cast<__nv_bfloat16*>(tensor) + index, *reinterpret_cast<__nv_bfloat16*>(&value));
#endif
  }
#endif
}


template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if_t<!std::is_same_v<c10::Half, scalar_t> && !std::is_same_v<c10::BFloat16, scalar_t>>* =
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

#undef ATOMICADD
#undef NATIVE_ZERO_BF16

} // namespace at::native
