#pragma once
#include <ATen/NumericUtils.h>
#include <stdint.h>
#include <aten/src/ATen/cpu/vec256/vec256.h>


template <typename scalar_t>
static bool gt_or_nan(scalar_t x, scalar_t y) {
  return ((at::_isnan<scalar_t>(x) && !at::_isnan<scalar_t>(y)) || (x > y));
}

template <typename scalar_t, typename index_t, typename Comp, typename Fn>
index_t scalar_partition(
    scalar_t *arr,
    Comp gt_or_nan,
    Fn swap_fn,
    index_t L,
    index_t R,
    scalar_t piv) {
  index_t i = L, j = R;
  do {
    do
      i++;
    while (gt_or_nan(piv, arr[i]));
    do
      j--;
    while (gt_or_nan(arr[j], piv));
    if (j < i)
      break;
    swap_fn(i, j);
  } while (1);

  return j;
}

namespace at {
namespace native {

int32_t vec_qs_partition_inplace(float *begin, float *end,
    int32_t *indices,
    float pivot,
    bool largest);

}
}
