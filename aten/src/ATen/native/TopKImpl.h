#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/NumericUtils.h>

namespace at {
namespace native {

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

  using elem_t = std::pair<accscalar_t, int64_t>;
  std::vector<elem_t> queue(dim_size);
  for (const auto i : c10::irange(n)) {
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k, &mode_values_stride);
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k, &mode_indices_stride);
    TensorAccessor<scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size, &tmp_values_stride);

    auto n = dim_size;
    auto use_partial_sort = k * 64 <= n;

    for (const auto j : c10::irange(n)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    // we want nan to be sorted as top for numpy compatibility
    if (use_partial_sort) {
      if (largest) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
      } else {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
      }
    } else {
      if (largest) {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((_isnan<accscalar_t>(x.first) && !_isnan<accscalar_t>(y.first)) || (x.first > y.first));
            });
        }
      } else {
        std::nth_element(queue.begin(), queue.begin() + k -1, queue.end(),
          [](const elem_t& x, const elem_t& y) -> bool {
            return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
          });
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k -1,
            [](const elem_t& x, const elem_t& y) -> bool {
              return ((!_isnan<accscalar_t>(x.first) && _isnan<accscalar_t>(y.first)) || (x.first < y.first));
            });
        }
      }
    }

    for (const auto j : c10::irange(k)) {
      mode_values[j] = queue[j].first;
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace native
} // namespace at
