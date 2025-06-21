#pragma once

#include <atomic>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && CUDART_VERSION >= 12010
#define NVCC_SUPPORTS_MULTICAST 1
#endif

#include <ATen/ATen.h>
#if defined(USE_ROCM)
#include <hip/hip_bf16.h>
#endif
#if !defined(USE_ROCM)
#include <cuda_bf16.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
#include <cuda/atomic>
#endif
#endif
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace c10d::symmetric_memory {

template <int Size>
using Vec = at::native::memory::Vec<Size>;

template <class... T>
inline constexpr bool dependent_false =
    at::native::memory::dependent_false<T...>;

using at::native::memory::get_alignment;

template <std::memory_order Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t* addr, uint32_t compare, uint32_t val) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  ref.compare_exchange_strong(compare, val, ::cuda::std::memory_order(Sem));
  return compare;
#elif defined(USE_ROCM)
  __atomic_compare_exchange_n(
      addr, &compare, val, false, static_cast<int>(Sem), __ATOMIC_RELAXED);
  return compare;
#else
  CUDA_KERNEL_ASSERT(false);
  return 0;
#endif
}

__device__ __forceinline__ void trap() {
#if defined(USE_ROCM)
  // abort() calls trap() under the covers. However, on ROCm, the trap is
  // handled differently inside hip runtime. It collects a gpu core dump and
  // causes linux kernel to create a core dump of the host application.
  abort();
#else
  __trap();
#endif
}

__device__ __forceinline__ size_t global_timer_ns() {
#if defined(USE_ROCM)
  static constexpr double MI300_FREQ_GHZ = 2.1;
  return clock64() / MI300_FREQ_GHZ;
#else
  size_t val;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(val) : : "memory");
  return val;
#endif
}

constexpr size_t ns_per_ms = 1e6;

template <std::memory_order Sem>
__device__ __forceinline__ bool try_put_signal(
    uint32_t* addr,
    size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 0, 1) != 0) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}

template <std::memory_order Sem>
__device__ __forceinline__ bool try_wait_signal(
    uint32_t* addr,
    size_t timeout_ms) {
  size_t deadline = global_timer_ns() + timeout_ms * ns_per_ms;
  while (cas<Sem>(addr, 1, 0) != 1) {
    if (timeout_ms != 0 && global_timer_ns() > deadline) {
      return false;
    }
  }
  return true;
}

template <std::memory_order Sem>
__device__ __forceinline__ void put_signal(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <std::memory_order Sem>
__device__ __forceinline__ void wait_signal(uint32_t* addr) {
  while (cas<Sem>(addr, 1, 0) != 1)
    ;
}

// Synchronizes blocks with matching blockIdx across participating devices.
// Note: sync_remote_block itself is not a system level barrier/fence. It is a
// building block for expressing different synchronization patterns.
//
// Pattern 0: Ensures that all writes to symm_mem buffers from previous
// kernels across all devices are visible to the current kernel:
//
//   sync_remote_blocks<std::memory_order_relaxed>(...);
//   __syncthreads();
//
// Pattern 1: Ensures that all writes to symm_mem buffers from the current
// block are visible to all remote blocks with matching blockIdx:
//
//   __syncthreads();
//   sync_remote_blocks<std::memory_order_acq_rel>(...);
//   __syncthreads();
//
// Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
// for writing by subsequent kernels across all devices.
//
//   __syncthreads();
//   sync_remote_blocks<std::memory_order_relaxed>(...);
template <std::memory_order Sem>
__device__ __forceinline__ void sync_remote_blocks(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size);

template <>
__device__ __forceinline__ void sync_remote_blocks<std::memory_order_relaxed>(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    put_signal<std::memory_order_relaxed>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<std::memory_order_relaxed>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <>
__device__ __forceinline__ void sync_remote_blocks<std::memory_order_acq_rel>(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    put_signal<std::memory_order_release>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<std::memory_order_acquire>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <typename T>
struct MultimemLdReduce {
  template <int Alignment>
  __device__ __inline__ Vec<Alignment> operator()(T* mc_ptr) {
    static_assert(dependent_false<T>);
  }
};

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> multimem_ld_reduce_add(T* mc_ptr) {
  MultimemLdReduce<T> functor;
  return functor.template operator()<Alignment>(mc_ptr);
}

#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec) \
  template <>                                                          \
  struct MultimemLdReduce<type> {                                      \
    template <int Alignment>                                           \
    __device__ __inline__ Vec<Alignment> operator()(type* mc_ptr) {    \
      CUDA_KERNEL_ASSERT(false);                                       \
    }                                                                  \
  };
#else
#define SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(type, asm_type, acc_prec)    \
  template <>                                                             \
  struct MultimemLdReduce<type> {                                         \
    template <int Alignment>                                              \
    __device__ __inline__ Vec<Alignment> operator()(type* mc_ptr) {       \
      Vec<Alignment> vec;                                                 \
      if constexpr (Alignment == 16) {                                    \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v4" asm_type " {%0,%1,%2,%3}, [%4];"                        \
            : "=r"(vec.u32[0]),                                           \
              "=r"(vec.u32[1]),                                           \
              "=r"(vec.u32[2]),                                           \
              "=r"(vec.u32[3])                                            \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      } else if constexpr (Alignment == 8) {                              \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec          \
            ".v2" asm_type " {%0,%1}, [%2];"                              \
            : "=r"(vec.u32[0]), "=r"(vec.u32[1])                          \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      } else if constexpr (Alignment == 4) {                              \
        asm("multimem.ld_reduce.relaxed.sys.global.add" acc_prec asm_type \
            " %0, [%1];"                                                  \
            : "=r"(vec.u32)                                               \
            : "l"(mc_ptr)                                                 \
            : "memory");                                                  \
      }                                                                   \
      return vec;                                                         \
    }                                                                     \
  };
#endif

SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(at::BFloat16, ".bf16x2", ".acc::f32");
SPECIALIZE_MULTIMEM_LD_REDUCE_VEC_32(float, ".f32", "");

template <int Alignment, typename T>
__device__ __inline__ void multimem_st(T* mc_ptr, Vec<Alignment>& vec) {
#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
  CUDA_KERNEL_ASSERT(false);
#else
  if constexpr (Alignment == 16) {
    asm("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(mc_ptr),
          "r"(vec.u32[0]),
          "r"(vec.u32[1]),
          "r"(vec.u32[2]),
          "r"(vec.u32[3])
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};"
        :
        : "l"(mc_ptr), "r"(vec.u32[0]), "r"(vec.u32[1])
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("multimem.st.relaxed.sys.global.f32 [%0], %1;"
        :
        : "l"(mc_ptr), "r"(vec.u32)
        : "memory");
  } else {
    static_assert(dependent_false<T>);
  }
#endif
}

template <typename T>
__device__ __inline__ T add_bf16x2(T a, T b) {
  static_assert(sizeof(T) == 4);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
  return T{};
#else
  auto res = __hadd2(
      *reinterpret_cast<__nv_bfloat162*>(&a),
      *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<T*>(&res);
#endif
}

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> add_vec(
    const Vec<Alignment>& a,
    const Vec<Alignment>& b) {
  Vec<Alignment> c{};
  if constexpr (std::is_same_v<T, float>) {
    if constexpr (Alignment == 16) {
      c.f32[0] = a.f32[0] + b.f32[0];
      c.f32[1] = a.f32[1] + b.f32[1];
      c.f32[2] = a.f32[2] + b.f32[2];
      c.f32[3] = a.f32[3] + b.f32[3];
    } else if constexpr (Alignment == 8) {
      c.f32[0] = a.f32[0] + b.f32[0];
      c.f32[1] = a.f32[1] + b.f32[1];
    } else if constexpr (Alignment == 4) {
      c.f32 = a.f32 + b.f32;
    } else {
      static_assert(dependent_false<T>);
    }
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    if constexpr (Alignment == 16) {
      c.u32[0] = add_bf16x2(a.u32[0], b.u32[0]);
      c.u32[1] = add_bf16x2(a.u32[1], b.u32[1]);
      c.u32[2] = add_bf16x2(a.u32[2], b.u32[2]);
      c.u32[3] = add_bf16x2(a.u32[3], b.u32[3]);
    } else if constexpr (Alignment == 8) {
      c.u32[0] = add_bf16x2(a.u32[0], b.u32[0]);
      c.u32[1] = add_bf16x2(a.u32[1], b.u32[1]);
    } else if constexpr (Alignment == 4) {
      c.u32 = add_bf16x2(a.u32, b.u32);
    } else {
      static_assert(dependent_false<T>);
    }
  } else {
    static_assert(dependent_false<T>);
  }
  return c;
}

// With world_size specialization: perform balanced load from all peers before
// performing reduction.
template <typename T, int alignment, int k_world_size>
__device__ inline std::enable_if_t<(k_world_size > 0), Vec<alignment>>
load_and_reduce(T** ptrs, size_t rank, size_t world_size, size_t offset) {
  Vec<alignment> vecs[k_world_size];
#pragma unroll k_world_size
  for (size_t step = 0; step < k_world_size; ++step) {
    size_t remote_rank = (rank + step) % k_world_size;
    vecs[remote_rank] =
        at::native::memory::ld_vec<alignment>(ptrs[remote_rank] + offset);
  }
  auto acc = vecs[0];
#pragma unroll k_world_size - 1
  for (size_t r = 1; r < world_size; ++r) {
    acc = add_vec<alignment, T>(acc, vecs[r]);
  }
  return acc;
}

// Without world_size specialization: perform ordered (unbalanced) load and
// accumulate on each load.
template <typename T, int alignment, int k_world_size>
__device__ inline std::enable_if_t<(k_world_size <= 0), Vec<alignment>>
load_and_reduce(T** ptrs, size_t rank, size_t world_size, size_t offset) {
  Vec<alignment> acc{};
  for (size_t step = 0; step < world_size; ++step) {
    auto vec = at::native::memory::ld_vec<alignment>(ptrs[step] + offset);
    acc = add_vec<alignment, T>(acc, vec);
  }
  return acc;
}

} // namespace c10d::symmetric_memory
