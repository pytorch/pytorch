#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <numeric>

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

  // Fast path: unit-stride input, selecting via partial_sort. Sorting an
  // index vector (8 bytes/element) instead of (value, index) pairs
  // (16 bytes/element) halves the working set, and partial_sort's tail
  // scan still reads tmp_values sequentially since idx[i] == i for any
  // position not yet touched by the heap. Restricted to unit stride,
  // because a non-unit stride turns tmp_values[idx[j]] into a scattered
  // gather; restricted to partial_sort, because nth_element's swaps
  // scramble idx and remove the sequential-read property this relies on.
  if (tmp_values_stride == 1 && k * 64 <= dim_size) {
    static thread_local std::vector<int64_t> idx;
    idx.resize(dim_size);
    for (const auto i : c10::irange(n)) {
      TensorAccessor<scalar_t, 1> mode_values(
          reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
          &k, &mode_values_stride);
      TensorAccessor<int64_t, 1> mode_indices(
          reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
          &k, &mode_indices_stride);
      const scalar_t* tmp_values =
          reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);

      std::iota(idx.begin(), idx.end(), int64_t{0});

      // we want nan to be sorted as top for numpy compatibility
      if (largest) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
          [tmp_values](int64_t a, int64_t b) -> bool {
            accscalar_t va = tmp_values[a], vb = tmp_values[b];
            return (_isnan<accscalar_t>(va) && !_isnan<accscalar_t>(vb)) || (va > vb);
          });
      } else {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
          [tmp_values](int64_t a, int64_t b) -> bool {
            accscalar_t va = tmp_values[a], vb = tmp_values[b];
            return (!_isnan<accscalar_t>(va) && _isnan<accscalar_t>(vb)) || (va < vb);
          });
      }

      for (const auto j : c10::irange(k)) {
        mode_values[j] = tmp_values[idx[j]];
        mode_indices[j] = idx[j];
      }
    }
    return;
  }

  using elem_t = std::pair<accscalar_t, int64_t>;
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
} // namespace at::native
