#pragma once
#include <ATen/core/TensorAccessor.h>
#include <torch/headeronly/cuda/Atomic.h>
#include <torch/headeronly/cuda/KernelUtils.h>

#if !(defined(USE_ROCM) || ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))))
#include <cuda_bf16.h>
#endif

#if defined(USE_ROCM)
#include <device_functions.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <torch/headeronly/cuda/detail/ROCmMacros.h>
#endif

namespace at::native {

using torch::headeronly::fastAtomicAdd;
using torch::headeronly::fastSpecializedAtomicAdd;

// Scatter into a packed accessor without naming a bound.
//
// fastAtomicAdd needs to know how far it may look for a pairing partner -- see
// Note [Passing pointer and offset to fastAtomicAdd] in GridSampler.cu. A
// caller holding a raw pointer supplies that by hand, and too large a value
// lets the pairing write past the allocation. An accessor already carries what
// it would be derived from:
//
//   memory_span = 1 + sum over d of (size[d] - 1) * stride[d]
//
// the distance to the last addressable element rather than the element count,
// which is what at::native::storage_size_for computes on the host. The two
// differ for a strided view, whose gaps are legal pairing partners: inside the
// region, though belonging to whatever the view was taken from, and only ever
// receiving a zero.
template <
    typename scalar_t,
    size_t N,
    template <typename U> class PtrTraits,
    typename index_t,
    typename... Idx>
__device__ __forceinline__ void fastAtomicAdd(
    at::GenericPackedTensorAccessor<scalar_t, N, PtrTraits, index_t> accessor,
    scalar_t value,
    Idx... indices) {
  static_assert(
      sizeof...(Idx) == N, "fastAtomicAdd needs one index per dimension");
  const index_t index[N] = {static_cast<index_t>(indices)...};
  index_t offset = 0;
  index_t memory_span = 1;
#pragma unroll
  for (size_t d = 0; d < N; ++d) {
    offset += index[d] * accessor.stride(d);
    memory_span += (accessor.size(d) - 1) * accessor.stride(d);
  }
  fastAtomicAdd(accessor.data(), offset, memory_span, value, true);
}

__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t h,
    const size_t w) {
  return (nc * height + h) * width + w;
}

// for channels-last
template <typename index_t = size_t>
__device__ __forceinline__ index_t idx_cl(
    const index_t n,
    const index_t h,
    const index_t w,
    const index_t c,
    const index_t height,
    const index_t width,
    const index_t channel) {
  return ((n * height + h) * width + w) * channel + c;
}

#ifdef USE_ROCM
// This function implements a committed store.
// Upon returning, the store is committed to global memory.
// This is useful in avoiding the need for fences.
// If multiple stores are done in a row there is option to skip
// waiting for commit for all but the last store.
template <typename T, bool wait_for_commit = true>
__device__ inline void cmtdStore(void* address, T value) {
  int constexpr num_long_per_val = sizeof(value) / sizeof(long);
  int constexpr num_int_per_val = sizeof(value) / sizeof(int);
  int constexpr num_short_per_val = sizeof(value) / sizeof(short);
  int constexpr num_char_per_val = sizeof(value) / sizeof(char);
  union pnr {
    T v;
    long l[num_long_per_val];
    int i[num_int_per_val];
    short s[num_short_per_val];
    char c[num_char_per_val];
  } _pnr = {.v = value};
  if constexpr (num_long_per_val * sizeof(long) == sizeof(value))
    for (int i = 0; i < num_long_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<long*>(address) + i,
          _pnr.l[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_int_per_val * sizeof(int) == sizeof(value))
    for (int i = 0; i < num_int_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<int*>(address) + i,
          _pnr.i[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_short_per_val * sizeof(short) == sizeof(value))
    for (int i = 0; i < num_short_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<short*>(address) + i,
          _pnr.s[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  else if constexpr (num_char_per_val * sizeof(char) == sizeof(value))
    for (int i = 0; i < num_char_per_val; i++)
      __hip_atomic_store(
          reinterpret_cast<char*>(address) + i,
          _pnr.c[i],
          __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_AGENT);
  if constexpr (wait_for_commit) {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#if defined(__GFX12__)
    asm volatile("s_wait_storecnt(0)" ::: "memory");
#elif defined(__GFX10__) || defined(__GFX11__)
    asm volatile("s_waitcnt_vscnt null, 0" ::: "memory");
#else
    // Older architectures have only 'vmcnt' counter.
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
  }
}

// This function implements warp-level opportunistic fastatomics
// To reduce contention on an atomicAdd, this replaces per-thread atomicAdd with
// a per-warp atomicAdd. We identify all the threads within a warp that will
// perform an atomicAdd on the same destination address and perform the addition
// on the CU. Each warp elects a leader thread which does the atomicAdd to the
// destination address.
template <class scalar_t, class index_t>
__device__ __forceinline__ void opportunistic_fastAtomicAdd(
    scalar_t* self_ptr,
    index_t index,
    const index_t numel,
    scalar_t value) {
  // TODO: move the builtin checks around the point-of-use, and implement a
  //       fallback for their absence, so as to allow targets that implement a
  //       subset to derive some benefit at least
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_mov_dpp) ||
      !__builtin_amdgcn_is_invocable(
          __builtin_amdgcn_flat_atomic_fadd_v2bf16) ||
      !__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_v2f16))
    return;
  scalar_t* dst = self_ptr + index;

  // pack coalesced bf16 and fp16
  if constexpr (
      std::is_same<scalar_t, c10::BFloat16>::value ||
      std::is_same<scalar_t, c10::Half>::value) {
    typedef unsigned short __attribute__((ext_vector_type(2))) vec_short2;
    union ill {
      unsigned int i[2];
      int64_t il;
    };
    ill iil_, ill_oneUpDst = {};
    iil_.il = (int64_t)dst;
    ill_oneUpDst.i[0] = __builtin_amdgcn_mov_dpp(iil_.i[0], 0x130, 0xf, 0xf, 0);
    ill_oneUpDst.i[1] = __builtin_amdgcn_mov_dpp(iil_.i[1], 0x130, 0xf, 0xf, 0);
    union bfi {
      scalar_t bf;
      short s;
    } bfi_ = {.bf = value};
    bfi bfi_oneUpVal;

    bfi_oneUpVal.s = __builtin_amdgcn_mov_dpp(bfi_.s, 0x130, 0xf, 0xf, 0);
    auto oneUpVal = bfi_oneUpVal.bf;

    __half* target_addr = reinterpret_cast<__half*>(self_ptr + index);
    bool low_byte =
        (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);
    bool canCombnUp = (bool)(__activemask() & (1 << (threadIdx.x + 1))) &&
        (low_byte && index < (numel - 1)) &&
        (ill_oneUpDst.il - reinterpret_cast<int64_t>(dst) == sizeof(scalar_t));
    bool canCombnDn =
        (__builtin_amdgcn_mov_dpp(canCombnUp, 0x138, 0xf, 0xf, 0));

    if (__lane_id() % 2 == 0) {
      if (canCombnUp) {
        typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
        union bfvs {
          scalar_t bf[2];
          vec_short2 vs2;
          vec_fp162 df16;
        };
        bfvs bfvs_ = {};
        bfvs_.bf[0] = value;
        bfvs_.bf[1] = oneUpVal;
        if constexpr (std::is_same<scalar_t, c10::BFloat16>::value)
          __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)dst, bfvs_.vs2);
        else
          __builtin_amdgcn_flat_atomic_fadd_v2f16((__half2*)dst, bfvs_.df16);
        return;
      }
    } else {
      if (canCombnDn)
        return;
    }
  }

  // not coalesced, so now let try to capture lane-matches...

  if (numel > 16 /*<-hueristic threshold*/ * 64) {
    // well shucks, unlikely to capture same-dest atomics in a wave.
    // fall back to direct fastAtomic...
    fastAtomicAdd(self_ptr, index, numel, value, true);
    return;
  }

  // __activemask() -- finds the set of threads in the warp that are about to
  // perform atomicAdd
  // __match_any_sync() -- returns bit mask of the threads that have same dest
  // addr
  auto mask = __match_any_sync(__activemask(), (int64_t)dst);

  // select a leader thread
  int leader = __ffsll(mask) - 1;

  scalar_t crnt_val = (scalar_t)0;
  auto crnt_msk = mask >> (leader);
  int crnt_idx = leader;

  // __shfl is limited in the dtypes it accepts
  // That's why, we need these if/else to correctly do the addition on the CU
  if constexpr (sizeof(scalar_t) <= sizeof(int)) {
    union punner {
      int l;
      scalar_t s;
    };
    punner pnr = {};
    pnr.s = value;
    while (crnt_msk != 0) {
      if (crnt_msk & 1) {
        punner add_val = {};
        add_val.l = __shfl(pnr.l, crnt_idx);
        crnt_val += add_val.s;
      }
      crnt_idx++;
      crnt_msk = crnt_msk >> 1;
    }
  } else if constexpr (sizeof(scalar_t) <= sizeof(long)) {
    union punner {
      long l;
      scalar_t s;
    };
    punner pnr = {};
    pnr.s = value;
    while (crnt_msk != 0) {
      if (crnt_msk & 1) {
        punner add_val = {};
        add_val.l = __shfl(pnr.l, crnt_idx);
        crnt_val += add_val.s;
      }
      crnt_idx++;
      crnt_msk = crnt_msk >> 1;
    }
  } else if constexpr (sizeof(scalar_t) <= sizeof(long long)) {
    union punner {
      long long l;
      scalar_t s;
    };
    punner pnr = {};
    pnr.s = value;
    while (crnt_msk != 0) {
      if (crnt_msk & 1) {
        punner add_val = {};
        add_val.l = __shfl(pnr.l, crnt_idx);
        crnt_val += add_val.s;
      }
      crnt_idx++;
      crnt_msk = crnt_msk >> 1;
    }
  } else {
    union punner {
      long long l[2];
      scalar_t s;
    };
    punner pnr = {};
    pnr.s = value;
    while (crnt_msk != 0) {
      if (crnt_msk & 1) {
        punner add_val = {};
        add_val.l[0] = __shfl(pnr.l[0], crnt_idx);
        add_val.l[1] = __shfl(pnr.l[1], crnt_idx);
        crnt_val += add_val.s;
      }
      crnt_idx++;
      crnt_msk = crnt_msk >> 1;
    }
  }

  // Once the correct crnt_val is determined, only the leader thread does the
  // update to the dest addr
  if (__lane_id() == leader) {
    fastAtomicAdd(self_ptr, index, numel, crnt_val, true);
  }
}
#endif

} // namespace at::native
