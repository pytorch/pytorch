#pragma once

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && CUDART_VERSION >= 12010
#define NVCC_SUPPORTS_MULTICAST 1
#endif

#include <ATen/ATen.h>
#if !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif
namespace c10d::symmetric_memory {

template <typename T>
__inline__ size_t get_alignment(T ptr_or_size) {
  auto val = reinterpret_cast<uintptr_t>(ptr_or_size);
  if (val % 16 == 0) {
    return 16;
  } else if (val % 8 == 0) {
    return 8;
  } else if (val % 4 == 0) {
    return 4;
  } else if (val % 2 == 0) {
    return 2;
  } else {
    return 1;
  }
}

template <>
__inline__ size_t get_alignment<size_t>(size_t size) {
  return get_alignment(reinterpret_cast<void*>(size));
}

template <bool Value, class... Args>
inline constexpr bool dependent_bool_value = Value;

template <class... Args>
inline constexpr bool dependent_false = dependent_bool_value<false, Args...>;

template <auto... Args>
inline constexpr bool dependent_false_nt =
    dependent_bool_value<false, decltype(Args)...>;

enum class MemOpSem {
  Relaxed,
  Acquire,
  Release,
  AcqRel,
};

#define CAS_ASM(addr, compare, val, old_val, sem)                 \
  asm volatile("atom.global" sem ".sys.cas.b32 %0, [%1], %2, %3;" \
               : "=r"(old_val)                                    \
               : "l"(addr), "r"(compare), "r"(val)                \
               : "memory");

template <MemOpSem Sem>
__device__ __forceinline__ uint32_t
cas(uint32_t* addr, uint32_t compare, uint32_t val) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  uint32_t old_val;
  if constexpr (Sem == MemOpSem::Relaxed) {
    CAS_ASM(addr, compare, val, old_val, ".relaxed");
  } else if constexpr (Sem == MemOpSem::Acquire) {
    CAS_ASM(addr, compare, val, old_val, ".acquire");
  } else if constexpr (Sem == MemOpSem::Release) {
    CAS_ASM(addr, compare, val, old_val, ".release");
  } else {
    static_assert(dependent_false_nt<Sem>);
  }
  return old_val;
#endif
}

__device__ __forceinline__ void trap() {
#if defined(USE_ROCM)
  assert(0);
#else
  __trap();
#endif
}

__device__ __forceinline__ size_t global_timer_ns() {
#if defined(USE_ROCM)
  CUDA_KERNEL_ASSERT(false);
  return 0;
#else
  size_t val;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(val) : : "memory");
  return val;
#endif
}

constexpr size_t ns_per_ms = 1e6;

template <MemOpSem Sem>
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

template <MemOpSem Sem>
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

template <MemOpSem Sem>
__device__ __forceinline__ void put_signal(uint32_t* addr) {
  while (cas<Sem>(addr, 0, 1) != 0)
    ;
}

template <MemOpSem Sem>
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
//   sync_remote_blocks<MemOpSem::Relaxed>(...);
//   __syncthreads();
//
// Pattern 1: Ensures that all writes to symm_mem buffers from the current
// block are visible to all remote blocks with matching blockIdx:
//
//   __syncthreads();
//   sync_remote_blocks<MemOpSem::AcqRel>(...);
//   __syncthreads();
//
// Pattern 2: Ensures that symm_mem buffers read by the current kernel are safe
// for writing by subsequent kernels across all devices.
//
//   __syncthreads();
//   sync_remote_blocks<MemOpSem::Relaxed>(...);
template <MemOpSem Sem>
__device__ __forceinline__ void sync_remote_blocks(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size);

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::Relaxed>(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Relaxed>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Relaxed>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <>
__device__ __forceinline__ void sync_remote_blocks<MemOpSem::AcqRel>(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size) {
  if (threadIdx.x < world_size) {
    auto target_rank = threadIdx.x;
    put_signal<MemOpSem::Release>(
        signal_pads[target_rank] + blockIdx.x * world_size + rank);
    wait_signal<MemOpSem::Acquire>(
        signal_pads[rank] + blockIdx.x * world_size + target_rank);
  }
}

template <int Size>
union Vec;

template <>
union Vec<4> {
  uint16_t u16[2];
  uint32_t u32, as_scalar;
  float f32;
};

template <>
union Vec<8> {
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64, as_scalar;
  float f32[2];
};

template <>
union alignas(16) Vec<16> {
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  uint4 u128, as_scalar;
  float f32[4];
};

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

template <int Alignment, typename T>
__device__ __inline__ Vec<Alignment> ld_vec(const T* addr) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  Vec<Alignment> vec;
  if constexpr (Alignment == 16) {
    asm("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1]), "=r"(vec.u32[2]), "=r"(vec.u32[3])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("ld.global.v2.u32 {%0,%1}, [%2];"
        : "=r"(vec.u32[0]), "=r"(vec.u32[1])
        : "l"(addr)
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("ld.global.u32 %0, [%1];" : "=r"(vec.u32) : "l"(addr) : "memory");
  } else {
    static_assert(dependent_false<T>);
  }
  return vec;
#endif
}

template <int Alignment, typename T>
__device__ __inline__ void st_vec(T* addr, const Vec<Alignment>& vec) {
#if defined(USE_ROCM) || !defined(NVCC_SUPPORTS_MULTICAST)
  CUDA_KERNEL_ASSERT(false);
#else
  if constexpr (Alignment == 16) {
    asm("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
        :
        : "l"(addr),
          "r"(vec.u32[0]),
          "r"(vec.u32[1]),
          "r"(vec.u32[2]),
          "r"(vec.u32[3])
        : "memory");
  } else if constexpr (Alignment == 8) {
    asm("st.global.v2.u32 [%0], {%1,%2};"
        :
        : "l"(addr), "r"(vec.u32[0]), "r"(vec.u32[1])
        : "memory");
  } else if constexpr (Alignment == 4) {
    asm("st.global.u32 [%0], %1;" : : "l"(addr), "r"(vec.u32) : "memory");
  } else {
    static_assert(dependent_false<T>);
  }
#endif
}

#if defined(USE_ROCM)
using __nv_bfloat162 = uint32_t;
#endif

template <typename T>
__device__ __inline__ T add_bf16x2(T a, T b) {
  static_assert(sizeof(T) == 4);
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
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
    vecs[remote_rank] = ld_vec<alignment>(ptrs[remote_rank] + offset);
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
    auto vec = ld_vec<alignment>(ptrs[step] + offset);
    acc = add_vec<alignment, T>(acc, vec);
  }
  return acc;
}

} // namespace c10d::symmetric_memory
