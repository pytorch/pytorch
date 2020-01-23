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
// each scalar type. But instead, we chose an existing builtin vector type
// of the same size, and reinterpret the memory as the desired type.

template<int size>
struct size_to_backing_type;

template<>
struct size_to_backing_type<1> {
  using type1 = char1;
  using type2 = char2;
  using type4 = char4;
};

template<>
struct size_to_backing_type<2> {
  using type1 = short1;
  using type2 = short2;
  using type4 = short4;
};

template<>
struct size_to_backing_type<4> {
  using type1 = float1;
  using type2 = float2;
  using type4 = float4;
};

template<>
struct size_to_backing_type<8> {
  using type1 = double1;
  using type2 = double2;
  using type4 = double4;
};

template<>
struct size_to_backing_type<16> {
  struct type1 {
    double2 x;
  };
  struct type2 {
    double2 x, y;
  };
  struct type4 {
    double2 x, y, z, w;
  };
};

// reinterpret_cast does not always succeed. For example, if
// you want to cast from BFloat16 to int16_t, the compiler would
// think you are doing stupid thing and reject your code. But
// by design we need the compiler to completely ignore the type
// system and do the cast. That's why we have a typeless_cast.
template<typename to_type, typename from_type>
__device__ inline to_type typeless_cast(from_type value) {
  return *reinterpret_cast<to_type *>(&value);
}

template<int scalar_size, int vec_size>
struct vector {};

template<int scalar_size>
struct vector<scalar_size, 1> {
  using type = typename size_to_backing_type<scalar_size>::type1;
};

template<int scalar_size>
struct vector<scalar_size, 2> {
  using type = typename size_to_backing_type<scalar_size>::type2;
};

template<int scalar_size>
struct vector<scalar_size, 4> {
  using type = typename size_to_backing_type<scalar_size>::type4;
};

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

  template <int vec_size>  // vec_size: number of scalars, can be 1, 2, or 4.
  struct vectorized : public common {

    static_assert(thread_work_size_ % vec_size == 0, "The workload per thread must be a multiple of vec_size");
    static constexpr int loop_size = thread_work_size_ / vec_size;

    template<typename accessor_t, typename scalar_t>
    __device__ inline void load(accessor_t to, scalar_t *from) {
      using vec_t = typename vector<sizeof(scalar_t), vec_size>::type;
      vec_t *from_ = reinterpret_cast<vec_t *>(from);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        vec_t v = from_[index];
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
          to(vec_size * i + j) = typeless_cast<scalar_t>((&v.x)[j]);
        }
      }
    }

    template<typename accessor_t, typename scalar_t>
    __device__ inline void store(scalar_t *to, accessor_t from) {
      using vec_t = typename vector<sizeof(scalar_t), vec_size>::type;
      vec_t *to_ = reinterpret_cast<vec_t *>(to);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        vec_t v;
        for (int j = 0; j < vec_size; j++) {
          (&v.x)[j] = typeless_cast<decltype(v.x)>(from(vec_size * i + j));
        }
        to_[index] = v;
      }
    }
  };
};

template<typename scalar_t>
constexpr inline int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment = std::alignment_of<typename size_to_backing_type<sizeof(scalar_t)>::type2>::value;
  constexpr int vec4_alignment = std::alignment_of<typename size_to_backing_type<sizeof(scalar_t)>::type4>::value;
  if (address % vec4_alignment == 0) {
    return 4;
  } else if (address % vec2_alignment == 0) {
    return 2;
  }
  return 1;
}

}}} // namespace at::native::memory
