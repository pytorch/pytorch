#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/util/Exception.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

namespace {

// The builtin vector types of CUDA are very limited. There are types,
// for example bool, half, that does not have corresponding vector types.
// But these types also benefit from vectorized memory access. To make all
// the types vectorizable, we are not using the corresponding vector type for
// each scalar type. But instead, we chose an existing builtin type
// of the same size, and reinterpret the memory as the desired type.

template<int size>
struct backing_type {
  static_assert(size % sizeof(double2) == 0, "size must be a multiple of sizeof(double2)");
  struct type {
    double2 v[size / sizeof(double2)];
  };
};

template<> struct backing_type<1> { using type = char; };
template<> struct backing_type<2> { using type = char2; };
template<> struct backing_type<4> { using type = char4; };
template<> struct backing_type<8> { using type = double; };
template<> struct backing_type<16> { using type = double2; };

}  // namespace

template <
  int num_threads_,        // number of threads in a block.
  int thread_work_size_    // number of elements each block needs to handle.
>
struct policies {

  struct common {
    static constexpr int num_threads = num_threads_;
    static constexpr int thread_work_size = thread_work_size_;
    static constexpr int block_work_size = thread_work_size_ * num_threads_;
  };

  struct checked_unroll : public common {

    int remaining;

    __device__ checked_unroll(int remaining): remaining(remaining) {}

    template<typename accessor_t, typename scalar_t>
    __device__ inline void load(accessor_t to, scalar_t *from) {
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < thread_work_size_; i++) {
        if (thread_idx >= remaining) {
          return;
        }
        to(i) = from[thread_idx];
        thread_idx += num_threads_;
      }
    }

    template<typename accessor_t, typename scalar_t>
    __device__ inline void store(scalar_t *to, accessor_t from) {
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < thread_work_size_; i++) {
        if (thread_idx >= remaining) {
          return;
        }
        to[thread_idx] = from(i);
        thread_idx += num_threads_;
      }
    }
  };

  // Functions here does not do boundary check. It assumes the whole block
  // has its job to do. So the reminders should be handled by the the caller
  // manually.

  template <int vec_size>  // vec_size: number of scalars, can be 1, 2, or 3.
  struct vectorized : public common {
    static constexpr int loop_size = thread_work_size_ / vec_size;

    template<typename accessor_t, typename scalar_t>
    __device__ inline void load(accessor_t to, scalar_t *from) {
      using vec_t = typename backing_type<sizeof(scalar_t) * vec_size>::type;
      vec_t *from_ = reinterpret_cast<vec_t *>(from);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        union U { vec_t vector; scalar_t scalars[vec_size]; __device__ U(){}; } u;
        u.vector = from_[index];
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
          to(vec_size * i + j) = u.scalars[j];
        }
      }
    }

    template<typename accessor_t, typename scalar_t>
    __device__ inline void store(scalar_t *to, accessor_t from) {
      using vec_t = typename backing_type<sizeof(scalar_t) * vec_size>::type;
      vec_t *to_ = reinterpret_cast<vec_t *>(to);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        union U { vec_t vector; scalar_t scalars[vec_size]; __device__ U(){}; } u;
        for (int j = 0; j < vec_size; j++) {
          u.scalars[j] = from(vec_size * i + j);
        }
        to_[index] = u.vector;
      }
    }
  };
};

template<typename scalar_t>
inline int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  int vec1_alignment = std::alignment_of<typename backing_type<sizeof(scalar_t) * 1>::type>::value;
  int vec2_alignment = std::alignment_of<typename backing_type<sizeof(scalar_t) * 2>::type>::value;
  int vec4_alignment = std::alignment_of<typename backing_type<sizeof(scalar_t) * 4>::type>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  TORCH_INTERNAL_ASSERT(address % vec1_alignment == 0, "unaligned pointer");
  return 1;
}

}}} // namespace at::native::memory
