#pragma once

#include <stdint.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native { namespace {

using namespace vec256;

// all three operands contiguous
template <typename traits>
static inline bool is_binary_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
         strides[1] == sizeof(typename traits::arg1_t) &&
         strides[2] == sizeof(typename traits::arg2_t);
}

// arg1 is a scalar, output and arg2 are contiguous
template <typename traits>
static inline bool is_binary_contiguous_s1(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
         strides[1] == 0 &&
         strides[2] == sizeof(typename traits::arg2_t);
}

// arg2 is a scalar, output and arg1 are contiguous
template <typename traits>
static inline bool is_binary_contiguous_s2(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
         strides[1] == sizeof(typename traits::arg1_t) &&
         strides[2] == 0;
}

#define LOOP_HEADER(traits, data, strides) \
  using arg0_t = typename traits::result_type; \
  using arg1_t = typename traits::arg1_t; \
  using arg2_t = typename traits::arg2_t; \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1]; \
  const char* in2_ptr = data[2]; \
  int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];

#define VEC_LOOP_HEADER(traits, data, strides) \
  using scalar_t = typename traits::result_type; \
  using Vec = Vec256<scalar_t>; \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1]; \
  const char* in2_ptr = data[2]; \


// Basic loop binary operation (two inputs, one output). May be auto-vectorized
// by the compiler.
template <typename traits, typename func_t>
static inline void binary_loop(char** data, const int64_t* strides, int64_t i, int64_t n, func_t op) {
  LOOP_HEADER(traits, data, strides)
  for (; i < n; i++) {
    arg1_t in1 = *(arg1_t*)(in1_ptr + i * s1);
    arg2_t in2 = *(arg2_t*)(in2_ptr + i * s2);
    arg0_t out = op(in1, in2);
    *(arg0_t*)(out_ptr + i * s0) = out;
  }
}

template <typename traits, typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop(char** data, const int64_t* strides, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(traits, data, strides)
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) {
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(scalar_t));
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size) * sizeof(scalar_t));
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(scalar_t));
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size) * sizeof(scalar_t));
    auto out1 = vop(a1, b1);
    auto out2 = vop(a2, b2);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size) * sizeof(scalar_t));
  }
  binary_loop<traits>(data, strides, i, n, op);
}

template <typename traits, typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop_s1(char** data, const int64_t* strides, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(traits, data, strides)
  int64_t i = 0;
  auto a = Vec(*(scalar_t*)in1_ptr);
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) {
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(scalar_t));
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size) * sizeof(scalar_t));
    auto out1 = vop(a, b1);
    auto out2 = vop(a, b2);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size) * sizeof(scalar_t));
  }
  binary_loop<traits>(data, strides, i, n, op);
}

template <typename traits, typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop_s2(char** data, const int64_t* strides, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(traits, data, strides)
  int64_t i = 0;
  auto b = Vec(*(scalar_t*)in2_ptr);
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) {
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(scalar_t));
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size) * sizeof(scalar_t));
    auto out1 = vop(a1, b);
    auto out2 = vop(a2, b);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size) * sizeof(scalar_t));
  }
  binary_loop<traits>(data, strides, i, n, op);
}

template <typename func_t>
void binary_kernel(TensorIterator& iter, func_t op) {
  using traits = binary_function_traits<func_t>;

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    // Specializations to encourage auto-vectorization (trick from Numpy's loops.c.src)
    if (is_binary_contiguous<traits>(strides)) {
      binary_loop<traits>(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s1<traits>(strides)) {
      binary_loop<traits>(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s2<traits>(strides)) {
      binary_loop<traits>(data, strides, 0, n, op);
    } else {
      binary_loop<traits>(data, strides, 0, n, op);
    }
  });
}

template <typename func_t, typename vec_func_t>
void binary_kernel_vec(TensorIterator& iter, func_t op, vec_func_t vop) {
  using traits = binary_function_traits<func_t>;
  static_assert(
    std::is_same<typename traits::result_type, typename traits::arg1_t>::value,
    "all types must match");
  static_assert(
    std::is_same<typename traits::result_type, typename traits::arg2_t>::value,
    "all types must match");

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    if (is_binary_contiguous<traits>(strides)) {
      vectorized_binary_loop<traits>(data, strides, n, op, vop);
    } else if (is_binary_contiguous_s1<traits>(strides)) {
      vectorized_binary_loop_s1<traits>(data, strides, n, op, vop);
    } else if (is_binary_contiguous_s2<traits>(strides)) {
      vectorized_binary_loop_s2<traits>(data, strides, n, op, vop);
    } else {
      binary_loop<traits>(data, strides, 0, n, op);
    }
  });
}

}}}  // namespace at::native::<anonymous>
