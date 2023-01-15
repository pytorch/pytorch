#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec.h>

namespace at { namespace vec {

// BFloat16 specification
template <typename scalar_t> struct VecScalarType { using type = scalar_t; };
template <> struct VecScalarType<BFloat16> { using type = float; };

// This is different from at::acc_type since we only need to specialize BFloat16
template <typename scalar_t>
using vec_scalar_t = typename VecScalarType<scalar_t>::type;

// Note that we already have specialized member of Vectorized<scalar_t> for BFloat16
// so the following functions would run smoothly:
//   using Vec = Vectorized<BFloat16>;
//   Vec one = Vec(BFloat16(1));
//   vec::map([](Vec x) { return one / (one + x.exp()); }, y_ptr, x_ptr, N);
//
// Then why we still need to specialize "funtional"?
//   If we do specialization at Vectorized<> level, the above example would need 3 pairs of
//   conversion of bf16->fp32/fp32->bf16, each for ".exp()", "+" and "/".
//   If we do specialization at vec::map<>() level, we have only 1 pair of conversion
//   of bf16->fp32/fp32->bf16, for the input and output BFloat16 vector only.
//
// The following BFloat16 functionality will only do data type conversion for input
// and output vector (reduce functionality will only convert the final scalar back to bf16).
// Compared to Vectorized<> specialization,
//   1. better performance since we have less data type conversion;
//   2. less rounding error since immediate results are kept in fp32;
//   3. accumulation done on data type of fp32.
//
//  If you plan to extend this file, please ensure adding unit tests at
//    aten/src/ATen/test/vec_test_all_types.cpp
//
template <typename scalar_t = BFloat16, typename Op>
inline BFloat16 reduce_all(const Op& vec_fun, const BFloat16* data, int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  return vec_reduce_all<float>(vec_fun, acc_fvec0);
}

template <typename scalar_t = BFloat16, typename Op1, typename Op2>
inline std::pair<BFloat16, BFloat16> reduce2_all(const Op1& vec_fun1, const Op2& vec_fun2,
    const BFloat16* data, int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
      vec_reduce_all<float>(vec_fun1, acc1_fvec0),
      vec_reduce_all<float>(vec_fun2, acc2_fvec0));
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <typename scalar_t = BFloat16, typename MapOp, typename ReduceOp>
inline BFloat16 map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    const BFloat16* data3,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  return vec_reduce_all<float>(red_fun, acc_fvec0);
}

template <typename scalar_t = BFloat16, typename Op>
inline void map(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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

template <typename scalar_t = BFloat16, typename Op>
inline void map4(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data1,
    const BFloat16* input_data2,
    const BFloat16* input_data3,
    const BFloat16* input_data4,
    int64_t size) {
  using bVec = vec::Vectorized<BFloat16>;
  using fVec = vec::Vectorized<float>;
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
    bVec data4_bvec = bVec::loadu(input_data4 + d);
    fVec data4_fvec0, data4_fvec1;
    std::tie(data4_fvec0, data4_fvec1) = convert_bfloat16_float(data4_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0, data4_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
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
    bVec data4_bvec = bVec::loadu(input_data4 + d, size - d);
    fVec data4_fvec0, data4_fvec1;
    std::tie(data4_fvec0, data4_fvec1) = convert_bfloat16_float(data4_bvec);
    fVec output_fvec0 = vec_fun(data1_fvec0, data2_fvec0, data3_fvec0, data4_fvec0);
    fVec output_fvec1 = vec_fun(data1_fvec1, data2_fvec1, data3_fvec1, data4_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

}} // namespace at::vec
