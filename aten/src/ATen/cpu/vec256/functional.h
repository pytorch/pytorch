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
    scalar_t* data,
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

template <typename scalar_t, typename Op>
inline void map4(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    const scalar_t* input_data4,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec1 = Vec::loadu(input_data1 + d);
    Vec data_vec2 = Vec::loadu(input_data2 + d);
    Vec data_vec3 = Vec::loadu(input_data3 + d);
    Vec data_vec4 = Vec::loadu(input_data4 + d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec1 = Vec::loadu(input_data1 + d, size - d);
    Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
    Vec data_vec3 = Vec::loadu(input_data3 + d, size - d);
    Vec data_vec4 = Vec::loadu(input_data4 + d, size - d);
    Vec output_vec = vec_fun(data_vec1, data_vec2, data_vec3, data_vec4);
    output_vec.store(output_data + d, size - d);
  }
}

}} // namespace at::vec256
