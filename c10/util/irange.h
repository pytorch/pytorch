// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <c10/util/Exception.h>
#include <c10/util/TypeSafeSignMath.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <type_traits>

namespace c10 {

namespace detail {

template <
    typename I,
    bool one_sided = false,
    typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
struct integer_iterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = I;
  using difference_type = std::ptrdiff_t;
  using pointer = I*;
  using reference = I&;

  explicit integer_iterator(I value) : value(value) {}

  I operator*() const {
    return value;
  }

  I const* operator->() const {
    return &value;
  }

  integer_iterator& operator++() {
    ++value;
    return *this;
  }

  integer_iterator operator++(int) {
    const auto copy = *this;
    ++*this;
    return copy;
  }

  bool operator==(const integer_iterator& other) const {
    if constexpr (one_sided) {
      // Range-for loops' end test is `begin != end`, not `begin <
      // end`. To handle `c10::irange(n)` where n < 0 (which should be
      // empty), we just make `begin != end` fail whenever `end` is
      // negative.
      return is_negative(other.value) || value == other.value;
    } else {
      return value == other.value;
    }
  }

  bool operator!=(const integer_iterator& other) const {
    return !(*this == other);
  }

 protected:
  I value;
};

} // namespace detail

template <
    typename I,
    bool one_sided = false,
    typename std::enable_if<std::is_integral<I>::value, bool>::type = true>
struct integer_range {
 public:
  integer_range(I begin, I end) : begin_(begin), end_(end) {}
  using iterator = detail::integer_iterator<I, one_sided>;
  iterator begin() const {
    return begin_;
  }
  iterator end() const {
    return end_;
  }

 private:
  iterator begin_;
  iterator end_;
};

/// Creates an integer range for the half-open interval [begin, end)
/// If end<=begin, then the range is empty.
/// The range has the type of the `end` integer; `begin` integer is
/// cast to this type.
template <
    typename Integer1,
    typename Integer2,
    typename std::enable_if<std::is_integral<Integer1>::value, bool>::type =
        true,
    typename std::enable_if<std::is_integral<Integer2>::value, bool>::type =
        true>
integer_range<Integer2> irange(Integer1 begin, Integer2 end) {
  // If end<=begin then the range is empty; we can achieve this effect by
  // choosing the larger of {begin, end} as the loop terminator
  return {
      static_cast<Integer2>(begin),
      std::max(static_cast<Integer2>(begin), end)};
}

/// Creates an integer range for the half-open interval [0, end)
/// If end<=begin, then the range is empty
template <
    typename Integer,
    typename std::enable_if<std::is_integral<Integer>::value, bool>::type =
        true>
integer_range<Integer, true> irange(Integer end) {
  return {Integer(), end};
}

} // namespace c10
