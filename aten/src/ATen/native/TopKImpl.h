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

  // Comparator in output order. largest/has-NaN are passed as types, so each
  // instantiation folds down to a body with no branch of its own.
  // we want nan to be sorted as top for numpy compatibility
  auto make_comp = [](auto largest_c, auto check_nan_c) {
    return [](const elem_t& x, const elem_t& y) -> bool {
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
    };
  };

  const bool use_heap = k * 64 <= dim_size;
  // Holds the k kept elements on the heap path, or the whole row otherwise.
  // Reused across topk_impl_loop calls on this thread (e.g. once per decoded
  // token) so a steady-state size doesn't pay a fresh malloc/free each time.
  static thread_local std::vector<elem_t> buf;
  buf.resize(use_heap ? k : dim_size);

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

    if (use_heap) {
      // Bounded size-k heap, rooted at the weakest of the k kept so far. Most
      // elements are rejected by a single compare, so the row is never copied.
      auto run = [&](auto largest_c) {
        constexpr bool kLargest = decltype(largest_c)::value;
        auto comp = make_comp(largest_c, std::true_type{});
        for (const auto j : c10::irange(k)) {
          buf[j] = elem_t(tmp_values[j], j);
        }
        std::make_heap(buf.begin(), buf.end(), comp);
        for (const auto j : c10::irange(k, dim_size)) {
          const accscalar_t v = tmp_values[j];
          const accscalar_t w = buf.front().first;
          // Cheap filter that never rejects a NaN on either side; those fall
          // through to the exact test, which applies the NaN ordering.
          if (kLargest ? !(v <= w) : !(w <= v)) {
            const elem_t cand(v, j);
            if (comp(cand, buf.front())) {
              std::pop_heap(buf.begin(), buf.end(), comp);
              buf.back() = cand;
              std::push_heap(buf.begin(), buf.end(), comp);
            }
          }
        }
        std::sort_heap(buf.begin(), buf.end(), comp);
      };
      if (largest) {
        run(std::true_type{});
      } else {
        run(std::false_type{});
      }
    } else {
      // NaN is rare in practice (e.g. model logits), so fold the check into
      // this already-required pass and let the comparator below skip it.
      // `|=` rather than `||`: the short-circuit form serializes the loop and
      // blocks vectorization of the copy.
      bool has_nan = false;
      for (const auto j : c10::irange(dim_size)) {
        buf[j].first = tmp_values[j];
        buf[j].second = j;
        has_nan |= _isnan<accscalar_t>(buf[j].first);
      }

      auto run = [&](auto largest_c, auto check_nan_c) {
        auto comp = make_comp(largest_c, check_nan_c);
        std::nth_element(buf.begin(), buf.begin() + k - 1, buf.end(), comp);
        if (sorted) {
          std::sort(buf.begin(), buf.begin() + k - 1, comp);
        }
      };

      if (largest && has_nan) {
        run(std::true_type{}, std::true_type{});
      } else if (largest) {
        run(std::true_type{}, std::false_type{});
      } else if (has_nan) {
        run(std::false_type{}, std::true_type{});
      } else {
        run(std::false_type{}, std::false_type{});
      }
    }

    for (const auto j : c10::irange(k)) {
      mode_values[j] = buf[j].first;
      mode_indices[j] = buf[j].second;
    }
  }
}

} // namespace CPU_CAPABILITY
} // namespace at::native
