#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace vec256 {

// TODO: Make this more efficient
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    vec256::Vec256<scalar_t> acc_vec,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  scalar_t acc_arr[Vec::size()];
  acc_vec.store(acc_arr);
  for (int64_t i = 1; i < size; i++) {
    std::array<scalar_t, Vec::size()> acc_arr_next = {0};
    acc_arr_next[0] = acc_arr[i];
    Vec acc_vec_next = Vec::loadu(acc_arr_next.data());
    acc_vec = vec_fun(acc_vec, acc_vec_next);
  }
  acc_vec.store(acc_arr);
  return acc_arr[0];
}

template <typename scalar_t, typename Op>
inline scalar_t reduce_all(const Op& vec_fun, const scalar_t* data, int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
  int64_t d = Vec::size();
  Vec acc_vec = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec = vec_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(vec_fun, acc_vec, Vec::size());
}

// similar to reduce_all, but reduces into two outputs
template <typename scalar_t, typename Op1, typename Op2>
inline std::pair<scalar_t, scalar_t> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const scalar_t* data, int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size()) {
    auto loaded_data = Vec::loadu(data, size);
    return std::pair<scalar_t, scalar_t>(
      vec_reduce_all(vec_fun1, loaded_data, size),
      vec_reduce_all(vec_fun2, loaded_data, size));
  }
  int64_t d = Vec::size();
  Vec acc_vec1 = Vec::loadu(data);
  Vec acc_vec2 = Vec::loadu(data);
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    acc_vec1 = vec_fun1(acc_vec1, data_vec);
    acc_vec2 = vec_fun2(acc_vec2, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    acc_vec1 = Vec::set(acc_vec1, vec_fun1(acc_vec1, data_vec), size - d);
    acc_vec2 = Vec::set(acc_vec2, vec_fun2(acc_vec2, data_vec), size - d);
  }
  return std::pair<scalar_t, scalar_t>(
    vec_reduce_all(vec_fun1, acc_vec1, Vec::size()),
    vec_reduce_all(vec_fun2, acc_vec2, Vec::size()));
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size())
    return vec_reduce_all(red_fun, map_fun(Vec::loadu(data, size)), size);
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    data_vec = map_fun(data_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    data_vec = map_fun(data_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec, Vec::size());
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    data_vec = map_fun(data_vec, data2_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }
  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec, Vec::size());
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    const scalar_t* data3,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size()) {
    Vec data_vec = Vec::loadu(data, size);
    Vec data2_vec = Vec::loadu(data2, size);
    Vec data3_vec = Vec::loadu(data3, size);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }

  int64_t d = Vec::size();
  Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(data + d);
    Vec data2_vec = Vec::loadu(data2 + d);
    Vec data3_vec = Vec::loadu(data3 + d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(data + d, size - d);
    Vec data2_vec = Vec::loadu(data2 + d, size - d);
    Vec data3_vec = Vec::loadu(data3 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec, data3_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec, Vec::size());
}

template <typename scalar_t, typename Op>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d));
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(input_data + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::loadu(input_data + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map3(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3);
    output_vec.store(output_data + d, size - d);
  }
}

// BFloat16 specification
template <typename scalar_t> struct VecScalarType { using type = scalar_t; };
template <> struct VecScalarType<BFloat16> { using type = float; };

// This is different from at::acc_type since we only need to specialize BFloat16
template <typename scalar_t>
using vec_scalar_t = typename VecScalarType<scalar_t>::type;

// Note that we already have specializes member of Vec256<scalar_t> for BFloat16
// so the following functionw would run smoothly:
//   using Vec = Vec256<BFloat16>;
//   Vec one = Vec(BFloat16(1));
//   vec256::map([](Vec x) { return one / (one + x.exp()); }, y_ptr, x_ptr, N);
//
// Why we still need to specializes "funtional"?
//   If we do specialization at Vec256<> level, the above example would need 3 pairs of
// conversion of bf16->fp32/fp32->bf16, each for ".exp()", "+" and "/".
//   If we do specialization at vec256::map<>() level, we have only 1 pair of conversion
// of bf16->fp32/fp32->bf16, for the input and output BFloat16 vector only.
//
// The following BFloat16 functionalities will only do data type conversion for input
// and output vector (reduce functionalities will only convert the final scalar back to bf16).
// Compared to Vec256<> specialization,
//   1. better performance since we have less data type conversion;
//   2. less rounding error since immediate results are kept in fp32;
//   3. accumulation done on data type of fp32.
//
//  If you plan to extend this file, make sure add unit test at
//    aten/src/ATen/test/vec256_test_all_types.cpp
//
template <typename scalar_t = BFloat16, typename Op>
inline BFloat16 reduce_all(const Op& vec_fun, const BFloat16* data, int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = fVec::set(data_fvec0, vec_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(vec_fun, data_fvec0, fVec::size());
    } else {
      return vec_reduce_all<float>(vec_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = vec_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, vec_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      acc_fvec0 = fVec::set(acc_fvec0, vec_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = vec_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(vec_fun, acc_fvec0, fVec::size());
}

template <typename scalar_t = BFloat16, typename Op1, typename Op2>
inline std::pair<BFloat16, BFloat16> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const BFloat16* data, int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      fVec acc1_fvec = fVec::set(data_fvec0, vec_fun1(data_fvec0, data_fvec1), size - fVec::size());
      fVec acc2_fvec = fVec::set(data_fvec0, vec_fun2(data_fvec0, data_fvec1), size - fVec::size());
      return std::pair<BFloat16, BFloat16>(
          vec_reduce_all<float>(vec_fun1, acc1_fvec, fVec::size()),
          vec_reduce_all<float>(vec_fun2, acc2_fvec, fVec::size()));
    } else {
      return std::pair<BFloat16, BFloat16>(
          vec_reduce_all<float>(vec_fun1, data_fvec0, size),
          vec_reduce_all<float>(vec_fun2, data_fvec0, size));
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc1_fvec0, acc1_fvec1;
  std::tie(acc1_fvec0, acc1_fvec1) = convert_bfloat16_float(acc_bvec);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
    acc1_fvec1 = vec_fun1(acc1_fvec1, data_fvec1);
    acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
    acc2_fvec1 = vec_fun2(acc2_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
      acc1_fvec1 = fVec::set(acc1_fvec1, vec_fun1(acc1_fvec1, data_fvec1), size - d - fVec::size());
      acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
      acc2_fvec1 = fVec::set(acc2_fvec1, vec_fun2(acc2_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      acc1_fvec0 = fVec::set(acc1_fvec0, vec_fun1(acc1_fvec0, data_fvec0), size - d);
      acc2_fvec0 = fVec::set(acc2_fvec0, vec_fun2(acc2_fvec0, data_fvec0), size - d);
    }
  }
  acc1_fvec0 = vec_fun1(acc1_fvec0, acc1_fvec1);
  acc2_fvec0 = vec_fun2(acc2_fvec0, acc2_fvec1);
  return std::pair<BFloat16, BFloat16>(
      vec_reduce_all<float>(vec_fun1, acc1_fvec0, fVec::size()),
      vec_reduce_all<float>(vec_fun2, acc2_fvec0, fVec::size()));
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  acc_fvec0 = map_fun(acc_fvec0);
  acc_fvec1 = map_fun(acc_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    data_fvec0 = map_fun(data_fvec0);
    data_fvec1 = map_fun(data_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc2_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    const BFloat16* data3,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3, size);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc2_bvec);
  bVec acc3_bvec = bVec::loadu(data3);
  fVec acc3_fvec0, acc3_fvec1;
  std::tie(acc3_fvec0, acc3_fvec1) = convert_bfloat16_float(acc3_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0, acc3_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1, acc3_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d, size - d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename scalar_t = BFloat16, typename Op>
inline void map(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename scalar_t = BFloat16, typename Op>
inline void map2(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    const BFloat16* input_data2,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename scalar_t = BFloat16, typename Op>
inline void map3(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data1,
    const BFloat16* input_data2,
    const BFloat16* input_data3,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data1_bvec = bVec::loadu(input_data1 + d);
    fVec data1_fvec0, data1_fvec1;
    std::tie(data1_fvec0, data1_fvec1) = convert_bfloat16_float(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data1_bvec = bVec::loadu(input_data1 + d, size - d);
    fVec data1_fvec0, data1_fvec1;
    std::tie(data1_fvec0, data1_fvec1) = convert_bfloat16_float(data1_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

}} // namespace at::vec256
