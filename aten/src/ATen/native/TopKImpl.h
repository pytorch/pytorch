#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/NumericUtils.h>
#include <type_traits>

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
  // Reused across topk_impl_loop calls on this thread (e.g. once per decoded
  // token) so a steady-state dim_size doesn't pay a fresh malloc/free each time.
  static thread_local std::vector<elem_t> queue;
  queue.resize(dim_size);
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

    // NaN is rare in practice (e.g. model logits), so fold the check into
    // this already-required pass and let every comparator below skip it.
    bool has_nan = false;
    for (const auto j : c10::irange(n_2)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
      has_nan = has_nan || _isnan<accscalar_t>(queue[j].first);
    }

    // we want nan to be sorted as top for numpy compatibility
    auto select = [&](auto comp) {
      if (use_partial_sort) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), comp);
      } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), comp);
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1, comp);
        }
      }
    };

    // largest/has_nan are passed as types, so each instantiation folds down to
    // a comparator with no branch of its own on the hot comparison path.
    auto select_as = [&](auto largest_c, auto check_nan_c) {
      select([](const elem_t& x, const elem_t& y) -> bool {
        constexpr bool kLargest = decltype(largest_c)::value;
        constexpr bool kCheckNan = decltype(check_nan_c)::value;
        // Ordering for `smallest` is the `largest` one with operands swapped.
        const elem_t& a = kLargest ? x : y;
        const elem_t& b = kLargest ? y : x;
        bool result = a.first > b.first;
        if constexpr (kCheckNan) {
          result = result ||
              (_isnan<accscalar_t>(a.first) && !_isnan<accscalar_t>(b.first));
        }
        return result;
      });
    };

    if (largest && has_nan) {
      select_as(std::true_type{}, std::true_type{});
    } else if (largest) {
      select_as(std::true_type{}, std::false_type{});
    } else if (has_nan) {
      select_as(std::false_type{}, std::true_type{});
    } else {
      select_as(std::false_type{}, std::false_type{});
    }

    for (const auto j : c10::irange(k)) {
      mode_values[j] = queue[j].first;
      mode_indices[j] = queue[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
