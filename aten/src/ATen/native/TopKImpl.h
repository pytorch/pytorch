#pragma once
#include <ATen/Context.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/NumericUtils.h>

namespace at::native {

#ifdef CPU_CAPABILITY
inline namespace CPU_CAPABILITY {
#else
inline namespace DEFAULT {
#endif

// Core topk loop, shared between CPU and QuantizedCPU
template <typename scalar_t, typename accscalar_t>
void topk_impl_loop(
    const int64_t mode_values_stride,
    const int64_t mode_indices_stride,
    const int64_t tmp_values_stride,
    const int64_t k,
    const int64_t dim_size,
    const bool largest,
    const bool sorted,
    char** data, const int64_t* strides, const int64_t n) {

  // If k is zero, then output values and indices are empty tensors
  // So iterating over other dims is pointless
  if (k == 0) {
    return;
  }
  using elem_t = std::pair<accscalar_t, int64_t>;
  // See Note [Enabling Deterministic Operations]
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  // Keep NumPy-compatible NaN ordering: NaNs sort first for largest=True and
  // last for largest=False.
  auto topk_comp = [largest, deterministic](const elem_t& x, const elem_t& y) -> bool {
    const bool x_nan = _isnan<accscalar_t>(x.first);
    const bool y_nan = _isnan<accscalar_t>(y.first);
    if (x_nan || y_nan) {
      if (x_nan != y_nan) {
        return largest ? x_nan : !x_nan;
      }
      return deterministic ? x.second < y.second : false;
    }
    if (x.first == y.first) {
      return deterministic ? x.second < y.second : false;
    }
    return largest ? x.first > y.first : x.first < y.first;
  };
  std::vector<elem_t> queue(dim_size);
  for (const auto i : c10::irange(n)) {
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k, &mode_values_stride);
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k, &mode_indices_stride);
    TensorAccessor<const scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size, &tmp_values_stride);

    auto n_2 = dim_size;
    auto use_partial_sort = k * 64 <= n_2;

    for (const auto j : c10::irange(n_2)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    if (use_partial_sort) {
      std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), topk_comp);
    } else {
      std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), topk_comp);
      if (sorted) {
        std::sort(queue.begin(), queue.begin() + k, topk_comp);
      }
    }

    for (const auto j : c10::irange(k)) {
      mode_values[j] = queue[j].first;
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
