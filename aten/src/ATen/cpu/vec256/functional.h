#pragma once
#include "vec256.h"

namespace at { namespace vec256 {
namespace {

template <int64_t size>
inline int64_t _leftover(int64_t x, int64_t y) {
  if (x + size > y)
    return y - x;
  return size;
}

} // namespace

template <typename scalar_t>
inline scalar_t reduce_all(
    Vec256<scalar_t> (*vec_fun)(Vec256<scalar_t>&, Vec256<scalar_t>&),
    scalar_t* data,
    int64_t size,
    scalar_t ident) {
  Vec256<scalar_t> acc_vec(ident);
  for (int64_t d = 0; d < size; d += Vec256<scalar_t>::size) {
    Vec256<scalar_t> value(ident);
    int64_t leftover = _leftover<Vec256<scalar_t>::size>(d, size);
    value.load(data + d, leftover);
    acc_vec = vec_fun(acc_vec, value);
  }
  scalar_t acc_arr[Vec256<scalar_t>::size];
  acc_vec.store(acc_arr);

  for (int64_t i = Vec256<scalar_t>::size / 2; i >= 1; i = i / 2) {
    scalar_t acc_arr_first[Vec256<scalar_t>::size];
    scalar_t acc_arr_second[Vec256<scalar_t>::size];
    for (int64_t j = 0; j < i; j++) {
      acc_arr_first[j] = acc_arr[j];
      acc_arr_second[j] = acc_arr[j + i];
    }
    Vec256<scalar_t> vec_first;
    Vec256<scalar_t> vec_second;
    vec_first.load(acc_arr_first, i);
    vec_second.load(acc_arr_second, i);
    acc_vec = vec_fun(vec_first, vec_second);
    acc_vec.store(acc_arr, i);
  }
  return acc_arr[0];
}

}} // namespace at::vec256
