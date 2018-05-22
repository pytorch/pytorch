#pragma once
#include "vec256.h"

namespace at { namespace vec256 {

// TODO: Make this more efficient
template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    vec256::Vec256<scalar_t> acc_vec,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  scalar_t acc_arr[Vec::size];
  acc_vec.store(acc_arr);
  for (int64_t i = 1; i < size; i++) {
    scalar_t acc_arr_next[Vec::size];
    acc_arr_next[0] = acc_arr[i];
    Vec acc_vec_next = Vec::load(acc_arr_next);
    acc_vec = vec_fun(acc_vec, acc_vec_next);
  }
  acc_vec.store(acc_arr);
  return acc_arr[0];
}

template <typename scalar_t, typename Op>
inline scalar_t reduce_all(const Op& vec_fun, scalar_t* data, int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size)
    return vec_reduce_all(vec_fun, Vec::load(data, size), size);
  int64_t d = Vec::size;
  Vec acc_vec = Vec::load(data);
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec data_vec = Vec::load(data + d);
    acc_vec = vec_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::load(data + d, size - d);
    acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(vec_fun, acc_vec, Vec::size);
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    scalar_t* data,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size)
    return vec_reduce_all(red_fun, map_fun(Vec::load(data, size)), size);
  int64_t d = Vec::size;
  Vec acc_vec = map_fun(Vec::load(data));
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec data_vec = Vec::load(data + d);
    data_vec = map_fun(data_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::load(data + d, size - d);
    data_vec = map_fun(data_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec, Vec::size);
}

template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    scalar_t* data,
    scalar_t* data2,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  if (size < Vec::size) {
    Vec data_vec = Vec::load(data, size);
    Vec data2_vec = Vec::load(data2, size);
    data_vec = map_fun(data_vec, data2_vec);
    return vec_reduce_all(red_fun, data_vec, size);
  }
  int64_t d = Vec::size;
  Vec acc_vec = map_fun(Vec::load(data), Vec::load(data2));
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec data_vec = Vec::load(data + d);
    Vec data2_vec = Vec::load(data2 + d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = red_fun(acc_vec, data_vec);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::load(data + d, size - d);
    Vec data2_vec = Vec::load(data2 + d, size - d);
    data_vec = map_fun(data_vec, data2_vec);
    acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
  }
  return vec_reduce_all(red_fun, acc_vec, Vec::size);
}

template <typename scalar_t, typename Op>
inline void map_(
    const Op& vec_fun,
    scalar_t* data,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec output_vec = vec_fun(Vec::load(data + d));
    output_vec.store(data + d);
  }
  if (size - d > 0) {
    Vec output_vec = vec_fun(Vec::load(data + d, size - d));
    output_vec.store(data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    scalar_t* input_data,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec output_vec = vec_fun(Vec::load(input_data + d));
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec output_vec = vec_fun(Vec::load(input_data + d, size - d));
    output_vec.store(output_data + d, size - d);
  }
}

template <typename scalar_t, typename Op>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    scalar_t* input_data,
    scalar_t* input_data2,
    int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size); d += Vec::size) {
    Vec data_vec = Vec::load(input_data + d);
    Vec data_vec2 = Vec::load(input_data2 + d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d);
  }
  if (size - d > 0) {
    Vec data_vec = Vec::load(input_data + d, size - d);
    Vec data_vec2 = Vec::load(input_data2 + d, size - d);
    Vec output_vec = vec_fun(data_vec, data_vec2);
    output_vec.store(output_data + d, size - d);
  }
}

}} // namespace at::vec256
