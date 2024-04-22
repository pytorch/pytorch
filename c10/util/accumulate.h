// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/ArrayRef.h>

#include <iterator>
#include <numeric>
#include <type_traits>

namespace c10 {

/// Sum of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t sum_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(), container.end(), static_cast<int64_t>(0));
}

/// Sum of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    typename std::enable_if<
        std::is_integral<
            typename std::iterator_traits<Iter>::value_type>::value,
        int>::type = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

/// Product of a list of integers; accumulates into the int64_t datatype
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t multiply_integers(const C& container) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

/// Product of integer elements referred to by iterators; accumulates into the
/// int64_t datatype
template <
    typename Iter,
    typename std::enable_if<
        std::is_integral<
            typename std::iterator_traits<Iter>::value_type>::value,
        int>::type = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

/// Return product of all dimensions starting from k
/// Returns 1 if k>=dims.size()
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(k >= 0);

  if (k > static_cast<int>(dims.size())) {
    return 1;
  } else {
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

/// Product of all dims up to k (not including dims[k])
/// Throws an error if k>dims.size()
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT((unsigned)k <= dims.size());

  auto cend = dims.cbegin();
  std::advance(cend, k);
  return multiply_integers(dims.cbegin(), cend);
}

/// Product of all dims between k and l (including dims[k] and excluding
/// dims[l]) k and l may be supplied in either order
template <
    typename C,
    typename std::enable_if<
        std::is_integral<typename C::value_type>::value,
        int>::type = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT(0 <= l);

  if (k > l) {
    std::swap(k, l);
  }

  TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  std::advance(cbegin, k);
  std::advance(cend, l);
  return multiply_integers(cbegin, cend);
}

} // namespace c10
