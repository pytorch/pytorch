#pragma once

#include <stdint.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native { namespace {

using namespace vec256;

// all three operands contiguous
static inline bool is_binary_contiguous(const int64_t* strides, const int64_t* element_sizes) {
  return strides[0] == element_sizes[0] &&
         strides[1] == element_sizes[1] &&
         strides[2] == element_sizes[2];
}

// arg1 is a scalar, output and arg2 are contiguous
static inline bool is_binary_contiguous_s1(const int64_t* strides, const int64_t* element_sizes) {
  return strides[0] == element_sizes[0] &&
         strides[1] == 0 &&
         strides[2] == element_sizes[2];
}

// arg2 is a scalar, output and arg1 are contiguous
static inline bool is_binary_contiguous_s2(const int64_t* strides, const int64_t* element_sizes) {
  return strides[0] == element_sizes[0] &&
         strides[1] == element_sizes[1] &&
         strides[2] == 0;
}

// Basic loop binary operation (two inputs, one output). May be auto-vectorized
// by the compiler.
#define BASE_BINARY_LOOP(op) \
  for (; i < n; i++) { \
    arg1_t in1 = *(arg1_t*)(in1_ptr + i * s1); \
    arg2_t in2 = *(arg2_t*)(in2_ptr + i * s2); \
    arg0_t out = op(in1, in2); \
    *(arg0_t*)(out_ptr + i * s0) = out; \
  }

#define VECTORIZED_BINARY_LOOP(vop, op) \
  using Vec = Vec256<arg0_t>; \
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) { \
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(arg0_t)); \
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size) * sizeof(arg0_t)); \
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(arg0_t)); \
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size) * sizeof(arg0_t)); \
    auto out1 = vop(a1, b1); \
    auto out2 = vop(a2, b2); \
    out1.store(out_ptr + i * sizeof(arg0_t)); \
    out2.store(out_ptr + (i + Vec::size) * sizeof(arg0_t)); \
  } \
  BASE_BINARY_LOOP(op)

#define VECTORIZED_BINARY_LOOP_S1(vop, op) \
  using Vec = Vec256<arg0_t>; \
  auto a = Vec(*(arg0_t*)in1_ptr); \
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) { \
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(arg0_t)); \
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size) * sizeof(arg0_t)); \
    auto out1 = vop(a, b1); \
    auto out2 = vop(a, b2); \
    out1.store(out_ptr + i * sizeof(arg0_t)); \
    out2.store(out_ptr + (i + Vec::size) * sizeof(arg0_t)); \
  } \
  BASE_BINARY_LOOP(op)

#define VECTORIZED_BINARY_LOOP_S2(vop, op) \
  using Vec = Vec256<arg0_t>; \
  auto b = Vec(*(arg0_t*)in2_ptr); \
  for (; i <= n - 2 * Vec::size; i += 2 * Vec::size) { \
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(arg0_t)); \
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size) * sizeof(arg0_t)); \
    auto out1 = vop(a1, b); \
    auto out2 = vop(a2, b); \
    out1.store(out_ptr + i * sizeof(arg0_t)); \
    out2.store(out_ptr + (i + Vec::size) * sizeof(arg0_t)); \
  } \
  BASE_BINARY_LOOP(op)


template <typename func_t>
void binary_kernel(TensorIterator& iter, func_t f) {
  using traits = binary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;
  using arg2_t = typename traits::arg2_t;

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    char* __restrict__ out_ptr = data[0];
    const char* __restrict__ in1_ptr = data[1];
    const char* __restrict__ in2_ptr = data[2];
    int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];
    int64_t i = 0;
    // Specializations to encourage auto-vectorization (trick from Numpy's loops.c.src)
    if (s0 == sizeof(arg0_t) && s1 == sizeof(arg1_t) && s2 == sizeof(arg2_t)) {
      BASE_BINARY_LOOP(f);
    } else if (s0 == sizeof(arg0_t) && s1 == 0 && s2 == sizeof(arg2_t)) {
      BASE_BINARY_LOOP(f);
    } else if (s0 == sizeof(arg0_t) && s1 == sizeof(arg1_t) && s2 == 0) {
      BASE_BINARY_LOOP(f);
    } else {
      BASE_BINARY_LOOP(f);
    }
  });
}

template <typename func_t, typename vec_func_t>
void binary_kernel_vec(TensorIterator& iter, func_t f, vec_func_t vf) {
  using traits = binary_function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  using arg1_t = typename traits::arg1_t;
  using arg2_t = typename traits::arg2_t;
  static_assert(std::is_same<arg0_t, arg1_t>::value, "all types must match");
  static_assert(std::is_same<arg0_t, arg2_t>::value, "all types must match");

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    char* __restrict__ out_ptr = data[0];
    const char* __restrict__ in1_ptr = data[1];
    const char* __restrict__ in2_ptr = data[2];
    int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];
    int64_t i = 0;
    if (s0 == sizeof(arg0_t) && s1 == sizeof(arg1_t) && s2 == sizeof(arg2_t)) {
      VECTORIZED_BINARY_LOOP(vf, f)
    } else if (s0 == sizeof(arg0_t) && s1 == 0 && s2 == sizeof(arg2_t)) {
      VECTORIZED_BINARY_LOOP_S1(vf, f);
    } else if (s0 == sizeof(arg0_t) && s1 == sizeof(arg1_t) && s2 == 0) {
      VECTORIZED_BINARY_LOOP_S2(vf, f);
    } else {
      BASE_BINARY_LOOP(f);
    }
  });
}

}}}  // namespace at::native::<anonymous>
