#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorAccessor.h>

namespace at::native {

#ifdef CPU_CAPABILITY
inline namespace CPU_CAPABILITY {
#else
inline namespace DEFAULT {
#endif

template <typename scalar_t, bool largest>
struct TopKComparator {
  bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    if constexpr (largest) {
      return (_isnan(lhs.first) && !_isnan(rhs.first)) ||
          (lhs.first > rhs.first);
    } else {
      return (!_isnan(lhs.first) && _isnan(rhs.first)) ||
          (lhs.first < rhs.first);
    }
  }
};

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
    char** data,
    const int64_t* strides,
    const int64_t n) {
  // If k is zero, then output values and indices are empty tensors
  // So iterating over other dims is pointless
  if (k == 0) {
    return;
  }
  using elem_t = std::pair<accscalar_t, int64_t>;

  const auto use_partial_sort = k * 64 <= dim_size;
  std::vector<elem_t> queue(use_partial_sort ? k : dim_size);
  for (const auto i : c10::irange(n)) {
    TensorAccessor<scalar_t, 1> mode_values(
        reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
        &k,
        &mode_values_stride);
    TensorAccessor<int64_t, 1> mode_indices(
        reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
        &k,
        &mode_indices_stride);
    TensorAccessor<const scalar_t, 1> tmp_values(
        reinterpret_cast<scalar_t*>(data[2] + i * strides[2]),
        &dim_size,
        &tmp_values_stride);

    for (const auto j : c10::irange(use_partial_sort ? k : dim_size)) {
      queue[j].first = tmp_values[j];
      queue[j].second = j;
    }

    // we want nan to be sorted as top for numpy compatibility
    if (use_partial_sort) {
      auto queue_begin = queue.begin();
      auto queue_end = queue.begin() + k;
      auto& queue_front_ref = queue.front();
      auto& queue_back_ref = queue.back();

      if (largest) {
        auto comp = TopKComparator<elem_t, true>{};
        std::make_heap(queue_begin, queue_end, comp);

        for (auto idx = k; idx < dim_size; ++idx) {
          elem_t this_val{tmp_values[idx], idx};
          if (comp(this_val, queue_front_ref)) {
            std::pop_heap(queue_begin, queue_end, comp);
            queue_back_ref = this_val;
            std::push_heap(queue_begin, queue_end, comp);
          }
        }

        if (sorted) {
          std::sort_heap(queue_begin, queue_end, comp);
        }
      } else {
        auto comp = TopKComparator<elem_t, false>{};
        std::make_heap(queue_begin, queue_end, comp);

        for (auto idx = k; idx < dim_size; ++idx) {
          elem_t this_val{tmp_values[idx], idx};
          if (comp(this_val, queue_front_ref)) {
            std::pop_heap(queue_begin, queue_end, comp);
            queue_back_ref = this_val;
            std::push_heap(queue_begin, queue_end, comp);
          }
        }

        if (sorted) {
          std::sort_heap(queue_begin, queue_end, comp);
        }
      }
    } else {
      if (largest) {
        auto comp = TopKComparator<elem_t, true>{};
        std::nth_element(
            queue.begin(), queue.begin() + k - 1, queue.end(), comp);
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1, comp);
        }
      } else {
        auto comp = TopKComparator<elem_t, false>{};
        std::nth_element(
            queue.begin(), queue.begin() + k - 1, queue.end(), comp);
        if (sorted) {
          std::sort(queue.begin(), queue.begin() + k - 1, comp);
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
