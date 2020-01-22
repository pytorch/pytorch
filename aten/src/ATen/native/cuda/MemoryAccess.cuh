#pragma once

#include <cstdint>
#include <type_traits>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

namespace {

template <typename scalar_t>
struct has_builtin_vector_type : public std::false_type {};

template <> struct has_builtin_vector_type<uint8_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int8_t>  : public std::true_type {};
template <> struct has_builtin_vector_type<int16_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int>     : public std::true_type {};
template <> struct has_builtin_vector_type<int64_t> : public std::true_type {};
template <> struct has_builtin_vector_type<float>   : public std::true_type {};
template <> struct has_builtin_vector_type<double>  : public std::true_type {};

// for types that does not have corresponding builtin vector type,
// it is ensured that dynamic dispatch will never use it. But
// we need to create a stub for it for completeness
template <typename scalar_t>
struct fake_vector {  // this is just a placeholder
  scalar_t x, y, z, w;
};

template <typename scalar_t, int size>
struct Info {
  static constexpr int alignment = sizeof(scalar_t);
  using vector_type = fake_vector<scalar_t>;
};

#define DEFINE_VECTOR_INFO(TYPE, SIZE, VECTYPE, ALIGNMENT)    \
  template <>                                                 \
  struct Info<TYPE, SIZE> {                                   \
    static constexpr int alignment = ALIGNMENT;               \
    using vector_type = VECTYPE;                              \
  }

// Note: alignment data could be found at:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types

//                    TYPE, SIZE,     VECTYPE, ALIGNMENT
DEFINE_VECTOR_INFO(uint8_t,    2,      uchar2,         2);
DEFINE_VECTOR_INFO(uint8_t,    4,      uchar4,         4);

DEFINE_VECTOR_INFO( int8_t,    2,       char2,         2);
DEFINE_VECTOR_INFO( int8_t,    4,       char4,         4);

DEFINE_VECTOR_INFO(int16_t,    2,      short2,         4);
DEFINE_VECTOR_INFO(int16_t,    4,      short4,         8);

DEFINE_VECTOR_INFO(    int,    2,        int2,         8);
DEFINE_VECTOR_INFO(    int,    4,        int4,        16);

static_assert(sizeof(int64_t) == sizeof(long long), "long long is assumed to be 64bit");
DEFINE_VECTOR_INFO(int64_t,    2,   longlong2,        16);
DEFINE_VECTOR_INFO(int64_t,    4,   longlong4,        16);

DEFINE_VECTOR_INFO(  float,    2,      float2,         8);
DEFINE_VECTOR_INFO(  float,    4,      float4,        16);

DEFINE_VECTOR_INFO( double,    2,     double2,        16);
DEFINE_VECTOR_INFO( double,    4,     double4,        16);

#undef DEFINE_VECTOR_INFO

template <typename scalar_t, int size>
struct Vec;

template <typename scalar_t>
struct Vec<scalar_t, 1> {
  scalar_t v;
  __device__ inline scalar_t get(int i) {
    return v;  // no boundary check here
  }
  __device__ inline void set(int i, scalar_t value) {
    v = value;  // no boundary check here
  }
};

template <typename scalar_t>
struct Vec<scalar_t, 2> {
  typename Info<scalar_t, 2>::vector_type v;
  __device__ inline scalar_t get(int i) {
    if (i == 0) {
      return v.x;
    }
    return v.y;  // no boundary check here
  }
  __device__ inline void set(int i, scalar_t value) {
    if (i == 0) {
      v.x = value;
    } else {
      v.y = value;
    }
  }
};

template <typename scalar_t>
struct Vec<scalar_t, 4> {
  typename Info<scalar_t, 4>::vector_type v;
  __device__ inline scalar_t get(int i) {
    switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    case 2:
      return v.z;
    }
    return v.w; // no boundary check here
  }
  __device__ inline void set(int i, scalar_t value) {
    switch (i) {
    case 0:
      v.x = value; break;
    case 1:
      v.y = value; break;
    case 2:
      v.z = value; break;
    }
    v.w = value; // no boundary check here
  }
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

  template <int vec_size>  // vec_size: vector size, can be 1, 2, or 3.
  struct vectorized : public common {
    static constexpr int loop_size = thread_work_size_ / vec_size;

    template<typename accessor_t, typename scalar_t>
    __device__ inline void load(accessor_t to, scalar_t *from) {
      using vec_t = Vec<scalar_t, vec_size>;
      vec_t *from_ = reinterpret_cast<vec_t *>(from);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        vec_t vector = from_[index];
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
          to(vec_size * i + j) = vector.get(j);
        }
      }
    }

    template<typename accessor_t, typename scalar_t>
    __device__ inline void store(scalar_t *to, accessor_t from) {
      using vec_t = Vec<scalar_t, vec_size>;
      vec_t *to_ = reinterpret_cast<vec_t *>(to);
      int thread_idx = threadIdx.x;
      #pragma unroll
      for (int i = 0; i < loop_size; i++) {
        int index = thread_idx + i * num_threads_;
        vec_t vector;
        for (int j = 0; j < vec_size; j++) {
          vector.set(j, from(vec_size * i + j));
        }
        to_[index] = vector;
      }
    }
  };
};

template<typename scalar_t>
inline int can_vectorize_up_to(char *pointer) {
  if (has_builtin_vector_type<scalar_t>::value) {
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    if (address % Info<scalar_t, 4>::alignment == 0) {
      return 4;
    } else if (address % Info<scalar_t, 2>::alignment == 0) {
      return 2;
    }
  }
  return 1;
}

}}} // namespace at::native::memory
