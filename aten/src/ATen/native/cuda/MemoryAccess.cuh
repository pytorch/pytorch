#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

namespace policies {

struct checked_unroll {

  int remaining;

  __device__ checked_unroll(int remaining): remaining(remaining) {}

  __device__ inline bool check_inbounds(int thread_work_elem) {
    return ((threadIdx.x  + thread_work_elem*num_threads) < remaining);
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      to(i) = from[thread_idx];
      thread_idx += num_threads;
    }
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void store(accessor_t from, scalar_t *to) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      to[thread_idx] = from(i);
      thread_idx += num_threads;
    }
  }
};

// Functions here does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the the caller
// manually.

template <int vec_size>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(thread_work_size % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = thread_work_size / vec_size;

  __device__ inline constexpr bool check_inbounds(int thread_work_elem) {
    return true;
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load(accessor_t to, scalar_t *from) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t v = from_[index];
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template<typename accessor_t, typename scalar_t>
  __device__ inline void store(accessor_t from, scalar_t *to) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from(vec_size * i + j);
      }
      to_[index] = v;
    }
  }
};

}  // namespace policies

// This is only used in host, but we will wrap this into some templates
// which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
// in order to compile
template<typename scalar_t>
inline C10_HOST_DEVICE int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

}}} // namespace at::native::memory
