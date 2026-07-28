#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstdint>
#include <type_traits>

// The contents are device-only and compile away in plain host translation
// units.
#if defined(__CUDACC__) || defined(__HIPCC__)

// Header-only fastAtomicAdd for libtorch-agnostic (stable ABI) CUDA and HIP
// extensions, ported from ATen/native/cuda/KernelUtils.cuh with the
// gpuAtomicAddNoReturn fallbacks (ATen/cuda/Atomic.cuh) inlined as
// detail::atomicAddScalar. This file is consumed un-hipified, so ROCm paths
// are spelled with HIP-native types behind USE_ROCM.

#if defined(USE_ROCM)
#include <device_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

// Same fallbacks as ATen/cuda/detail/ROCmMacros.cuh, which is not reachable
// header-only.
#if !__has_builtin(__builtin_amdgcn_processor_is)
#if defined(__amdgcn_processor__)
// Device pass: __amdgcn_processor__ is available
#define __builtin_amdgcn_processor_is(x) \
  (__builtin_strcmp(x, __amdgcn_processor__) == 0)
#else
// Host pass: define a no-op fallback so the macro always exists
#define __builtin_amdgcn_processor_is(x) false
#endif // defined(__amdgcn_processor__)
#endif // !__has_builtin(__builtin_amdgcn_processor_is)

#if !__has_builtin(__builtin_amdgcn_is_invocable)
#define __builtin_amdgcn_is_invocable(x) (__has_builtin(x))
#endif

// ROCm 6.3 is planned to have these functions, but until then here they are.
#if ROCM_VERSION < 60400
__device__ inline __hip_bfloat162 preview_unsafeAtomicAdd(
    __hip_bfloat162* address,
    __hip_bfloat162 value) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_v2bf16)) {
    typedef unsigned short __attribute__((ext_vector_type(2))) vec_short2;
    static_assert(sizeof(vec_short2) == sizeof(__hip_bfloat162_raw));
    union {
      __hip_bfloat162_raw bf162_raw;
      vec_short2 vs2;
    } u{static_cast<__hip_bfloat162_raw>(value)};
    u.vs2 =
        __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)address, u.vs2);
    return static_cast<__hip_bfloat162>(u.bf162_raw);
  } else {
    static_assert(sizeof(unsigned int) == sizeof(__hip_bfloat162_raw));
    union u_hold {
      __hip_bfloat162_raw h2r;
      unsigned int u32;
    };
    u_hold old_val, new_val;
    old_val.u32 = __hip_atomic_load(
        (unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    do {
      new_val.h2r = __hadd2(old_val.h2r, value);
    } while (!__hip_atomic_compare_exchange_strong(
        (unsigned int*)address,
        &old_val.u32,
        new_val.u32,
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_AGENT));
    return old_val.h2r;
  }
}

__device__ inline __half2 preview_unsafeAtomicAdd(
    __half2* address,
    __half2 value) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_v2f16)) {
    // The api expects an ext_vector_type of half
    typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
    static_assert(sizeof(vec_fp162) == sizeof(__half2_raw));
    union {
      __half2_raw h2r;
      vec_fp162 fp16;
    } u{static_cast<__half2_raw>(value)};
    u.fp16 =
        __builtin_amdgcn_flat_atomic_fadd_v2f16((vec_fp162*)address, u.fp16);
    return static_cast<__half2>(u.h2r);
  } else {
    static_assert(sizeof(__half2_raw) == sizeof(unsigned int));
    union u_hold {
      __half2_raw h2r;
      unsigned int u32;
    };
    u_hold old_val, new_val;
    old_val.u32 = __hip_atomic_load(
        (unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    do {
      new_val.h2r = __hadd2(old_val.h2r, value);
    } while (!__hip_atomic_compare_exchange_strong(
        (unsigned int*)address,
        &old_val.u32,
        new_val.u32,
        __ATOMIC_RELAXED,
        __ATOMIC_RELAXED,
        __HIP_MEMORY_SCOPE_AGENT));
    return old_val.h2r;
  }
}
#define ATOMICADD preview_unsafeAtomicAdd
#else
#define ATOMICADD unsafeAtomicAdd
#endif
#define NATIVE_ZERO_BF16 __float2bfloat16(0.0f)
#else
#if !(defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
#include <cuda_bf16.h>
#endif
#define ATOMICADD atomicAdd
#define NATIVE_ZERO_BF16 __int2bfloat16_rz(0)
#endif

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

namespace detail {

#if defined(USE_ROCM)
using bfloat16_t = __hip_bfloat16;
using bfloat16x2_t = __hip_bfloat162;
#else
using bfloat16_t = __nv_bfloat16;
using bfloat16x2_t = __nv_bfloat162;
#endif

// 16-bit atomic add through a 32-bit CAS loop, for Half/BFloat16 on targets
// without a native scalar atomic (ROCm, CUDA below sm_70/sm_80). Mirrors the
// AtomicFPOp loops in ATen/cuda/Atomic.cuh.
template <typename scalar_t>
__device__ __forceinline__ void atomicAdd16BitCAS(
    scalar_t* address,
    scalar_t value) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    scalar_t sum;
    sum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    sum = sum + value;
    old = (size_t)address & 2 ? (old & 0xffff) | (sum.x << 16)
                              : (old & 0xffff0000) | sum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// Scalar (non-vectored) atomic adds with the same semantics ATen gets from
// gpuAtomicAddNoReturn for these types.
__device__ __forceinline__ void atomicAddScalar(float* address, float value) {
#if defined(USE_ROCM)
  // See Note [HIP unsafeAtomicAdd] in ATen/cuda/Atomic.cuh: correct only on
  // GPU memory, which callers of these helpers must guarantee.
  if (__builtin_amdgcn_processor_is("gfx908"))
    return atomicAddNoRet(address, value);
  (void)unsafeAtomicAdd(address, value);
#else
  atomicAdd(address, value);
#endif
}

__device__ __forceinline__ void atomicAddScalar(double* address, double value) {
#if defined(USE_ROCM)
  (void)unsafeAtomicAdd(address, value);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  // atomicAdd on double needs sm_60; polyfill with the CAS loop
  // gpuAtomicAddNoReturn uses below that.
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(address, value);
#endif
}

__device__ __forceinline__ void atomicAddScalar(Half* address, Half value) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  atomicAdd16BitCAS(address, value);
#else
  atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(value));
#endif
}

__device__ __forceinline__ void atomicAddScalar(
    BFloat16* address,
    BFloat16 value) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  atomicAdd16BitCAS(address, value);
#else
  atomicAdd(
      reinterpret_cast<bfloat16_t*>(address),
      *reinterpret_cast<bfloat16_t*>(&value));
#endif
}

template <typename scalar_t>
__device__ __forceinline__ void atomicAddScalar(
    scalar_t* address,
    scalar_t value) {
  // Types with a native atomicAdd only; the exotic integer/complex types
  // gpuAtomicAddNoReturn also covers are out of scope here.
  atomicAdd(address, value);
}

} // namespace detail

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
    typename std::enable_if_t<std::is_same_v<Half, scalar_t>>* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  detail::atomicAddScalar(tensor + index, value);
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
    detail::atomicAddScalar(tensor + index, value);
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
    typename std::enable_if_t<std::is_same_v<BFloat16, scalar_t>>* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  detail::atomicAddScalar(tensor + index, value);
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32
  // bit aligned)
  detail::bfloat16_t* target_addr =
      reinterpret_cast<detail::bfloat16_t*>(tensor + index);
  bool low_byte =
      (reinterpret_cast<std::uintptr_t>(target_addr) %
           sizeof(detail::bfloat16x2_t) ==
       0);

  if (low_byte && index < (numel - 1)) {
    detail::bfloat16x2_t value2;
    value2.x = *reinterpret_cast<detail::bfloat16_t*>(&value);
    value2.y = NATIVE_ZERO_BF16;
    ATOMICADD(reinterpret_cast<detail::bfloat16x2_t*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    detail::bfloat16x2_t value2;
    value2.x = NATIVE_ZERO_BF16;
    value2.y = *reinterpret_cast<detail::bfloat16_t*>(&value);
    ATOMICADD(reinterpret_cast<detail::bfloat16x2_t*>(target_addr - 1), value2);

  } else {
#ifdef USE_ROCM
    detail::atomicAddScalar(tensor + index, value);
#else
    atomicAdd(
        reinterpret_cast<detail::bfloat16_t*>(tensor) + index,
        *reinterpret_cast<detail::bfloat16_t*>(&value));
#endif
  }
#endif
}

template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if_t<
        !std::is_same_v<Half, scalar_t> &&
        !std::is_same_v<BFloat16, scalar_t>>* = nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
  detail::atomicAddScalar(tensor + index, value);
}

// index must be in [0, numel); numel bounds the 16-bit pairing above. On ROCm
// the float/double paths use unsafeAtomicAdd:
// https://rocm.docs.amd.com/projects/HIP/en/latest/reference/cpp_language_extensions.html#unsafe-floating-point-atomic-operations
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
    detail::atomicAddScalar(tensor + index, value);
  }
}

HIDDEN_NAMESPACE_END(torch, headeronly)

#undef ATOMICADD
#undef NATIVE_ZERO_BF16

#endif // defined(__CUDACC__) || defined(__HIPCC__)
