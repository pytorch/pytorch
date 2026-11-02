#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/irange.h>
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

#include <cstdint>
#include <limits>
#include <numeric>

namespace at::native {

using torch::headeronly::fastAtomicAdd;
using torch::headeronly::fastSpecializedAtomicAdd;

// Note [fastAtomicAdd bounds for strided tensors]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// fastAtomicAdd may pair a half/bfloat16 update with tensor[index +/- 1], so
// its (tensor, index, numel) must describe memory that the destination owns.
// Kernels that address a destination by element offset from its data pointer
// get that from FastAtomicAddBounds, which views the destination as rows:
// row r owns offsets [r * row_stride, r * row_stride + row_numel).
//
// - Non-overlapping and dense: a single row covering all of it.
// - Rows of row_numel contiguous elements whose other strides are all
//   multiples of row_stride >= row_numel, e.g. a column slice of a contiguous
//   matrix: the row and column of an offset are its quotient and remainder,
//   and only elements of the same row are paired.
// - Anything else, e.g. x[::2]: row_numel is 0, which disables pairing.
struct FastAtomicAddBounds {
  int64_t row_stride;
  int64_t row_numel;

  // Column of the element at offset; pass (data + offset - column, column,
  // row_numel) to fastAtomicAdd.
  __device__ __forceinline__ int64_t column(int64_t offset) const {
    return offset < row_stride ? offset : offset % row_stride;
  }
};

inline FastAtomicAddBounds fast_atomic_add_bounds(const TensorBase& self) {
  constexpr int64_t kSingleRow = std::numeric_limits<int64_t>::max();
  if (self.is_non_overlapping_and_dense()) {
    return {kSingleRow, self.numel()};
  }
  int64_t row_numel = 0;
  int64_t row_stride = 0;
  for (const auto d : c10::irange(self.dim())) {
    if (self.size(d) == 1) {
      continue;
    }
    if (self.stride(d) == 1 && row_numel == 0) {
      row_numel = self.size(d);
    } else {
      row_stride = std::gcd(row_stride, self.stride(d));
    }
  }
  if (row_numel == 0 || row_stride < row_numel) {
    return {kSingleRow, 0};
  }
  return {row_stride, row_numel};
}

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
// differ for a strided view, whose gaps are inside the region but belong to
// whatever the view was taken from. Pairing adds -0.0 to them, which preserves
// non-NaN values but is still a write the view does not own, so only use this
// overload on tensors whose whole memory span is owned by the kernel, such as
// freshly allocated gradients.
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
    typedef short __attribute__((ext_vector_type(2))) vec_short2;
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
        (low_byte && index + 1 < numel) &&
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
