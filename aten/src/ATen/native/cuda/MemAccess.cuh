#pragma once

#include <cstdint>
#include <type_traits>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

template <typename scalar_t>
struct has_builtin_vector_type : public std::false_type {};

template <> struct has_builtin_vector_type<uint8_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int8_t>  : public std::true_type {};
template <> struct has_builtin_vector_type<int16_t> : public std::true_type {};
template <> struct has_builtin_vector_type<int>     : public std::true_type {};
template <> struct has_builtin_vector_type<int64_t> : public std::true_type {};
template <> struct has_builtin_vector_type<float>   : public std::true_type {};
template <> struct has_builtin_vector_type<double>  : public std::true_type {};

template <typename scalar_t, int size>
struct Info;

#define DEFINE_VECTOR_INFO(TYPE, SIZE, VECTYPE, ALIGNMENT)    \
  template <>                                                 \
  struct Info<TYPE, SIZE> {                                   \
    static constexpr int alignment = ALIGNMENT;               \
    using vector_type = VECTYPE;                              \
  }

// Note: alignment data could be found at:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types

//                    TYPE, SIZE, VECTYPE, ALIGNMENT
DEFINE_VECTOR_INFO(uint8_t,    1, uint8_t,         1);
DEFINE_VECTOR_INFO(uint8_t,    2,  uchar2,         2);
DEFINE_VECTOR_INFO(uint8_t,    4,  uchar4,         4);

DEFINE_VECTOR_INFO( int8_t,    1,  int8_t,         1);
DEFINE_VECTOR_INFO( int8_t,    2,   char2,         2);
DEFINE_VECTOR_INFO( int8_t,    4,   char4,         4);

DEFINE_VECTOR_INFO(int16_t,    1, int16_t,         2);
DEFINE_VECTOR_INFO(int16_t,    2,  short2,         4);
DEFINE_VECTOR_INFO(int16_t,    4,  short4,         8);

DEFINE_VECTOR_INFO(    int,    1,     int,         4);
DEFINE_VECTOR_INFO(    int,    2,    int2,         8);
DEFINE_VECTOR_INFO(    int,    4,    int4,        16);

static_assert(sizeof(long) == 8, "long is assumed to be 64bit");
DEFINE_VECTOR_INFO(int64_t,    1, int64_t,         8);
DEFINE_VECTOR_INFO(int64_t,    2,   long2,        16);
DEFINE_VECTOR_INFO(int64_t,    4,   long4,        16);

DEFINE_VECTOR_INFO(  float,    1,   float,         4);
DEFINE_VECTOR_INFO(  float,    2,  float2,         8);
DEFINE_VECTOR_INFO(  float,    4,  float4,        16);

DEFINE_VECTOR_INFO( double,    1,  double,         8);
DEFINE_VECTOR_INFO( double,    2, double2,        16);
DEFINE_VECTOR_INFO( double,    4, double4,        16);

#undef DEFINE_VECTOR_INFO

template <typename scalar_t, int size>
struct Vec {
  typename Info<scalar_t, size>::vector_type v;
  static_assert(size == 1 || size == 2 || size == 4);
  __device__ inline scalar_t &operator[](int i) = delete;
};

template <typename scalar_t>
struct Vec<scalar_t, 1> {
  typename Info<scalar_t, 1>::vector_type v;
  __device__ inline scalar_t &operator[](int i) {
    return v;  // no boundary check here
  }
};

template <typename scalar_t>
struct Vec<scalar_t, 2> {
  typename Info<scalar_t, 2>::vector_type v;
  __device__ inline scalar_t &operator[](int i) {
    switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    }
    return 0;  // no boundary check here
  }
};

template <typename scalar_t>
struct Vec<scalar_t, 4> {
  typename Info<scalar_t, 4>::vector_type v;
  __device__ inline scalar_t &operator[](int i) {
    switch (i) {
    case 0:
      return v.x;
    case 1:
      return v.y;
    case 2:
      return v.z;
    case 3:
      return v.t;
    }
    return 0;  // no boundary check here
  }
};

// Functions here does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the the caller
// manually.

template <
  typename scalar_t,     // type of data.
  int num_threads,       // number of threads in a block.
  int block_load_size,   // number of elements each block needs to handle.
  int vec_size           // vector size, can be 1, 2, or 3.
>
struct vectorized {

  static constexpr int thread_load_size = block_load_size / num_threads;
  static constexpr int loop_size = thread_load_size / vec_size;

  __device__ inline void load(scalar_t to[thread_load_size], scalar_t *from) {
    using vec_t = Vec<scalar_t, vec_size>;
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t vector = from_[index];
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to[vec_size * i + j] = vector[j];
      }
    }
  }

  __device__ void store(scalar_t *to, scalar_t from[thread_load_size]) {
    using vec_t = Vec<scalar_t, vec_size>;
    vec_t *to_ = reinterpret_cast<vec_t *>(to);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      vec_t vector;
      for (int j = 0; j < vec_size; j++) {
        vector[j] = from[vec_size * i + j];
      }
      *to_ = vector;
    }
  }
};

template<typename scalar_t>
inline int can_vectorize_up_to(char *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  if (has_builtin_vector_type<scalar_t>::value) {
    if (address % Info<scalar_t, 4>::alignment == 0) {
      return 4;
    } else if (address % Info<scalar_t, 2>::alignment == 0) {
      return 2;
    }
  }
  return 1;
}

}}} // namespace at::native::memory
