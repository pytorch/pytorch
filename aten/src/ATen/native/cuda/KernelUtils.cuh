#pragma once
#include <ATen/cuda/Atomic.cuh>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

// ROCm 6.3 is planned to have these functions, but until then here they are.
#if defined(USE_ROCM) && ROCM_VERSION >= 60201
#include <device_functions.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

__device__ inline __hip_bfloat162 preview_unsafeAtomicAdd(__hip_bfloat162* address, __hip_bfloat162 value) {
#if (defined(__gfx942__)) && \
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
#if (defined(__gfx942__)) && \
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

#if (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__))
// This function implements warp-level opportunistic fastatomics
// To reduce contention on an atomicAdd, this replaces per-thread atomicAdd with a per-warp atomicAdd.
// We identify all the threads within a warp that will perform an atomicAdd on the same destination
// address and perform the addition on the CU. Each warp elects a leader thread which does the
// atomicAdd to the destination address.
template <class scalar_t, class index_t>
__device__ __forceinline__ void opportunistic_fastAtomicAdd(
    scalar_t* self_ptr,
    index_t index,
    const index_t numel,
    scalar_t value) {

    scalar_t* dst = self_ptr + index;

    //pack coalseced bf16 and fp16
    if constexpr (std::is_same<scalar_t, c10::BFloat16>::value || std::is_same<scalar_t, c10::Half>::value)
    {
        typedef unsigned short __attribute__((ext_vector_type(2))) vec_short2;
        union ill { unsigned int i[2]; int64_t il; };
        ill iil_, ill_oneUpDst = {};
        iil_.il = (int64_t)dst;
        ill_oneUpDst.i[0] = __builtin_amdgcn_mov_dpp(iil_.i[0], 0x130, 0xf, 0xf, 0);
        ill_oneUpDst.i[1] = __builtin_amdgcn_mov_dpp(iil_.i[1], 0x130, 0xf, 0xf, 0);
        union bfi {scalar_t bf; short s; } bfi_ = { .bf = value  }; bfi bfi_oneUpVal;

        bfi_oneUpVal.s = __builtin_amdgcn_mov_dpp(bfi_.s, 0x130, 0xf, 0xf, 0);
        auto oneUpVal = bfi_oneUpVal.bf;

        __half* target_addr = reinterpret_cast<__half*>(self_ptr + index);
        bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);
        bool canCombnUp = (bool)(__activemask()&(1<<(threadIdx.x+1))) &&
                                 (low_byte && index < (numel - 1)) &&
                                 (ill_oneUpDst.il - reinterpret_cast<int64_t>(dst) == sizeof(scalar_t));
        bool canCombnDn = (__builtin_amdgcn_mov_dpp(canCombnUp, 0x138, 0xf, 0xf, 0));

        if (__lane_id()%2==0)
        {
          if (canCombnUp) {
            typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
            union bfvs { scalar_t bf[2]; vec_short2 vs2; vec_fp162 df16;  };
            bfvs bfvs_ = {};
            bfvs_.bf[0] = value;
            bfvs_.bf[1] = oneUpVal;
            if constexpr (std::is_same<scalar_t, c10::BFloat16>::value)
              __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)dst, bfvs_.vs2);
            else
              __builtin_amdgcn_flat_atomic_fadd_v2f16((__half2*)dst, bfvs_.df16);
            return;
          }
        }
        else
        {
          if (canCombnDn)
            return;
        }
    }

    // not coalsced, so now let try to capture lane-matches...
    // __activemask() -- finds the set of threads in the warp that are about to perform atomicAdd
    // __match_any_sync() -- returns bit mask of the threads that have same dest addr
    auto mask = __match_any_sync(__activemask(), (int64_t)dst);

    // select a leader thread
    int leader = __ffsll(mask) - 1;

    scalar_t crnt_val = (scalar_t)0;
    auto crnt_msk = mask >> (leader);
    int crnt_idx = leader;

    // __shfl is limited in the dtypes it accepts
    // That's why, we need these if/else to correctly do the addition on the CU
    if constexpr(sizeof(scalar_t) <= sizeof(int)) {
     union punner { int l; scalar_t s; };
     punner pnr = {};
     pnr.s = value;
     while (crnt_msk != 0) {
        if (crnt_msk & 1) {
            punner add_val = {};
            add_val.l = __shfl(pnr.l ,crnt_idx);
            crnt_val += add_val.s;
        }
        crnt_idx++;
        crnt_msk = crnt_msk >> 1;
     }
    }
    else if constexpr(sizeof(scalar_t) <= sizeof(long)) {
     union punner { long l; scalar_t s; };
     punner pnr = {};
     pnr.s = value;
     while (crnt_msk != 0) {
        if (crnt_msk & 1) {
            punner add_val = {};
            add_val.l = __shfl(pnr.l ,crnt_idx);
            crnt_val += add_val.s;
        }
        crnt_idx++;
        crnt_msk = crnt_msk >> 1;
     }
    }
    else if constexpr(sizeof(scalar_t) <= sizeof(long long)) {
     union punner { long long l; scalar_t s; };
     punner pnr = {};
     pnr.s = value;
     while (crnt_msk != 0) {
        if (crnt_msk & 1) {
            punner add_val = {};
            add_val.l = __shfl(pnr.l ,crnt_idx);
            crnt_val += add_val.s;
        }
        crnt_idx++;
        crnt_msk = crnt_msk >> 1;
     }
    }
    else {
     union punner { long long l[2]; scalar_t s; };
     punner pnr = {};
     pnr.s = value;
     while (crnt_msk != 0) {
        if (crnt_msk & 1) {
            punner add_val = {};
            add_val.l[0] = __shfl(pnr.l[0] ,crnt_idx);
            add_val.l[1] = __shfl(pnr.l[1] ,crnt_idx);
            crnt_val += add_val.s;
        }
        crnt_idx++;
        crnt_msk = crnt_msk >> 1;
     }
    }


    //Once the correct crnt_val is determined, only the leader thread does the update to the dest addr
    if (__lane_id() == leader) {
      fastAtomicAdd(self_ptr, index, numel, crnt_val, true);
    }
}
#endif

#undef ATOMICADD
#undef NATIVE_ZERO_BF16

} // namespace at::native
