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

// all two operands contiguous
template <typename traits>
static inline bool is_unary_contiguous(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
         strides[1] == sizeof(typename traits::arg1_t);
}

// output is contiguous, arg1 is scalar
template <typename traits>
static inline bool is_unary_contiguous_s1(const int64_t* strides) {
  return strides[0] == sizeof(typename traits::result_type) &&
         strides[1] == 0;
}

// result is
static inline bool is_reduction(char** data, const int64_t* strides) {
  return strides[0] == 0 &&
         strides[1] == 0 &&
         data[0] == data[1];
}

#define UNARY_LOOP_HEADER(func_t, data, strides) \
  using traits = unary_function_traits<func_t>; \
  using arg0_t = typename traits::result_type; \
  using arg1_t = typename traits::arg1_t; \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1]; \
  int64_t s0 = strides[0], s1 = strides[1];

#define UNARY_VEC_HEADER(func_t) \
  using traits = unary_function_traits<func_t>; \
  using scalar_t = typename traits::result_type; \
  using Vec = Vec256<scalar_t>;

#define UNARY_VEC_LOOP_HEADER(func_t, data) \
  UNARY_VEC_HEADER(func_t) \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1];

#define LOOP_HEADER(func_t, data, strides) \
  using traits = binary_function_traits<func_t>; \
  using arg0_t = typename traits::result_type; \
  using arg1_t = typename traits::arg1_t; \
  using arg2_t = typename traits::arg2_t; \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1]; \
  const char* in2_ptr = data[2]; \
  int64_t s0 = strides[0], s1 = strides[1], s2 = strides[2];

#define VEC_HEADER(func_t) \
  using traits = binary_function_traits<func_t>; \
  using scalar_t = typename traits::result_type; \
  using Vec = Vec256<scalar_t>;

#define VEC_LOOP_HEADER(func_t, data) \
  VEC_HEADER(func_t) \
  char* out_ptr = data[0]; \
  const char* in1_ptr = data[1]; \
  const char* in2_ptr = data[2];

// Basic loop unary operation (one input, one output). May be auto-vectorized
// by the compiler.
template <typename func_t>
static inline void unary_loop(char** data, const int64_t* strides, int64_t i, int64_t n, func_t op) {
  UNARY_LOOP_HEADER(func_t, data, strides)
  for (; i < n; i++) {
    arg1_t in1 = *(arg1_t*)(in1_ptr + i * s1);
    arg0_t out = op(in1);
    *(arg0_t*)(out_ptr + i * s0) = out;
  }
}

// computes out = op(in1)
template <typename func_t, typename vec_func_t>
static inline void vectorized_unary_loop(char** data, int64_t n, func_t op, vec_func_t vop) {
  UNARY_VEC_LOOP_HEADER(func_t, data)
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(scalar_t));
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size()) * sizeof(scalar_t));
    auto out1 = vop(a1);
    auto out2 = vop(a2);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size()) * sizeof(scalar_t));
  }
  int64_t strides[] = { sizeof(scalar_t), sizeof(scalar_t) };
  unary_loop(data, strides, i, n, op);
}

// Basic loop binary operation (two inputs, one output). May be auto-vectorized
// by the compiler.
template <typename func_t>
static inline void binary_loop(char** data, const int64_t* strides, int64_t i, int64_t n, func_t op) {
  LOOP_HEADER(func_t, data, strides)
  for (; i < n; i++) {
    arg1_t in1 = *(arg1_t*)(in1_ptr + i * s1);
    arg2_t in2 = *(arg2_t*)(in2_ptr + i * s2);
    arg0_t out = op(in1, in2);
    *(arg0_t*)(out_ptr + i * s0) = out;
  }
}

// computes out = op(in1, in2)
template <typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop(char** data, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)
  int64_t i = 0;
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(scalar_t));
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size()) * sizeof(scalar_t));
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(scalar_t));
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size()) * sizeof(scalar_t));
    auto out1 = vop(a1, b1);
    auto out2 = vop(a2, b2);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size()) * sizeof(scalar_t));
  }
  int64_t strides[] = { sizeof(scalar_t), sizeof(scalar_t), sizeof(scalar_t) };
  binary_loop(data, strides, i, n, op);
}

// computes out = op(in1, in2) where in1 is a constant
template <typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop_s1(char** data, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)
  int64_t i = 0;
  auto a = Vec(*(scalar_t*)in1_ptr);
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto b1 = Vec::loadu(in2_ptr + i * sizeof(scalar_t));
    auto b2 = Vec::loadu(in2_ptr + (i + Vec::size()) * sizeof(scalar_t));
    auto out1 = vop(a, b1);
    auto out2 = vop(a, b2);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size()) * sizeof(scalar_t));
  }
  int64_t strides[] = { sizeof(scalar_t), 0, sizeof(scalar_t) };
  binary_loop(data, strides, i, n, op);
}

// computes out = op(in1, in2) where in2 is a constant
template <typename func_t, typename vec_func_t>
static inline void vectorized_binary_loop_s2(char** data, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)
  int64_t i = 0;
  auto b = Vec(*(scalar_t*)in2_ptr);
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto a1 = Vec::loadu(in1_ptr + i * sizeof(scalar_t));
    auto a2 = Vec::loadu(in1_ptr + (i + Vec::size()) * sizeof(scalar_t));
    auto out1 = vop(a1, b);
    auto out2 = vop(a2, b);
    out1.store(out_ptr + i * sizeof(scalar_t));
    out2.store(out_ptr + (i + Vec::size()) * sizeof(scalar_t));
  }
  int64_t strides[] = { sizeof(scalar_t), sizeof(scalar_t), 0 };
  binary_loop(data, strides, i, n, op);
}

template <typename func_t, typename vec_func_t>
static inline void reduction128(char** data, int64_t n, int64_t stride, func_t op, vec_func_t vop, bool reduce) {
  VEC_HEADER(func_t)
  char* out_ptr = data[0];
  char* in_ptr = data[1];
  Vec acc[4];
  for  (int j = 0; j < 4; j++) {
    acc[j] = Vec::loadu(in_ptr + j * Vec::size() * sizeof(scalar_t));
  }
  for (int64_t i = 1; i < n; i++) {
    const char* ptr = in_ptr + stride * i;
    acc[0] = vop(acc[0], Vec::loadu(ptr + (0 * Vec::size() * sizeof(scalar_t))));
    acc[1] = vop(acc[1], Vec::loadu(ptr + (1 * Vec::size() * sizeof(scalar_t))));
    acc[2] = vop(acc[2], Vec::loadu(ptr + (2 * Vec::size() * sizeof(scalar_t))));
    acc[3] = vop(acc[3], Vec::loadu(ptr + (3 * Vec::size() * sizeof(scalar_t))));
  }
  if (reduce) {
    scalar_t buffer[Vec::size()];
    acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]));
    acc[0].store(buffer);
    for (int j = 1; j < Vec::size(); j++) {
      buffer[0] = op(buffer[0], buffer[j]);
    }
    auto dst = (scalar_t*)out_ptr;
    *dst = op(*dst, buffer[0]);
  } else {
    for (int j = 0; j < 4; j++) {
      auto dst = out_ptr + j * Vec::size() * sizeof(scalar_t);
      acc[j] = vop(acc[j], Vec::loadu(dst));
      acc[j].store(dst);
    }
  }
}

template <typename F>
static inline void UNARY_OUTER_LOOP(char* data[2], const int64_t strides[2], int64_t n, F f) {
  for (int j = 0; j < n; j++) {
    f();
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

// computes the reduction out = op(out, in)
template <typename func_t, typename vec_func_t>
static inline void vectorized_inner_reduction(char** data, int64_t n, func_t op, vec_func_t vop) {
  VEC_HEADER(func_t)
  int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t count = n / (4 * Vec::size());
  if (count > 0) {
    reduction128(data, count, vector_stride, op, vop, /*reduce=*/true);
  }
  char* ptrs[3] = { data[0], data[0], data[1] };
  int64_t strides[] = { 0, 0, sizeof(scalar_t) };
  binary_loop(ptrs, strides, count * 4 * Vec::size(), n, op);
}

// computes the reduction out = op(out, in)
template <typename func_t, typename vec_func_t>
static inline void vectorized_outer_reduction(char** data, int64_t inner_stride, int64_t size0, int64_t size1, func_t op, vec_func_t vop) {
  VEC_HEADER(func_t)

  // reduce down each column of 4 * Vec::size() elements (128 bytes)
  int64_t outer_stride[2] = { 128, 128 };
  UNARY_OUTER_LOOP(data, outer_stride, size1 / (4 * Vec::size()), [&] {
    reduction128(data, size0, inner_stride, op, vop, /*reduce=*/false);
  });

  // reduce down the remaining columns
  int64_t step[] = { sizeof(scalar_t), sizeof(scalar_t) };
  int64_t remaining = size1 % (4 * Vec::size());
  UNARY_OUTER_LOOP(data, step, remaining, [&] {
    char* ptrs[3] = { data[0], data[0], data[1] };
    int64_t strides[] = { 0, 0, inner_stride };
    binary_loop(ptrs, strides, 0, size0, op);
  });
}

template <typename func_t>
void unary_kernel(TensorIterator& iter, func_t op) {
  using traits = unary_function_traits<func_t>;

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    // Specializations to encourage auto-vectorization (trick from Numpy's loops.c.src)
    if (is_unary_contiguous<traits>(strides)) {
      unary_loop(data, strides, 0, n, op);
    } else if (is_unary_contiguous_s1<traits>(strides)) {
      unary_loop(data, strides, 0, n, op);
    } else {
      unary_loop(data, strides, 0, n, op);
    }
  });
}

template <typename func_t, typename vec_func_t>
void unary_kernel_vec(TensorIterator& iter, func_t op, vec_func_t vop) {
  using traits = unary_function_traits<func_t>;
  static_assert(
    std::is_same<typename traits::result_type, typename traits::arg1_t>::value,
    "all types must match");

  iter.for_each(
      [&](int ntensor, char** data, const int64_t* strides, int64_t n) {
        if (is_unary_contiguous<traits>(strides)) {
          vectorized_unary_loop(data, n, op, vop);
        } else if (is_unary_contiguous_s1<traits>(strides)) {
          unary_loop(data, strides, 0, n, op);
        } else {
          unary_loop(data, strides, 0, n, op);
        }
      });
}

template <typename func_t>
void binary_kernel(TensorIterator& iter, func_t op) {
  using traits = binary_function_traits<func_t>;

  iter.for_each([&](int ntensor, char** data, const int64_t* strides, int64_t n) {
    // Specializations to encourage auto-vectorization (trick from Numpy's loops.c.src)
    if (is_binary_contiguous<traits>(strides)) {
      binary_loop(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s1<traits>(strides)) {
      binary_loop(data, strides, 0, n, op);
    } else if (is_binary_contiguous_s2<traits>(strides)) {
      binary_loop(data, strides, 0, n, op);
    } else {
      binary_loop(data, strides, 0, n, op);
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
      vectorized_binary_loop(data, n, op, vop);
    } else if (is_binary_contiguous_s1<traits>(strides)) {
      vectorized_binary_loop_s1(data, n, op, vop);
    } else if (is_binary_contiguous_s2<traits>(strides)) {
      vectorized_binary_loop_s2(data, n, op, vop);
    } else {
      binary_loop(data, strides, 0, n, op);
    }
  });
}

}}}  // namespace at::native::<anonymous>
