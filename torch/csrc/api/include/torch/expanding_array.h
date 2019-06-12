#pragma once

#include <ATen/core/ArrayRef.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace torch {

/// A utility class that accepts either a container of `D`-many values, or a
/// single value, which is internally repeated `D` times. This is useful to
/// represent parameters that are multidimensional, but often equally sized in
/// all dimensions. For example, the kernel size of a 2D convolution has an `x`
/// and `y` length, but `x` and `y` are often equal. In such a case you could
/// just pass `3` to an `ExpandingArray<2>` and it would "expand" to `{3, 3}`.
template <size_t D, typename T = int64_t>
class ExpandingArray {
 public:
  /// Constructs an `ExpandingArray` from an `initializer_list`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::initializer_list<T> list)
      : ExpandingArray(std::vector<T>(list)) {}

  /// Constructs an `ExpandingArray` from a `vector`. The extent of the
  /// length is checked against the `ExpandingArray`'s extent parameter `D` at
  /// runtime.
  /*implicit*/ ExpandingArray(const std::vector<T>& values) {
    AT_CHECK(
        values.size() == D,
        "Expected ",
        D,
        " values, but instead got ",
        values.size());
    std::copy(values.begin(), values.end(), values_.begin());
  }

  /// Constructs an `ExpandingArray` from a single value, which is repeated `D`
  /// times (where `D` is the extent parameter of the `ExpandingArray`).
  /*implicit*/ ExpandingArray(T single_size) {
    values_.fill(single_size);
  }

  /// Constructs an `ExpandingArray` from a correctly sized `std::array`.
  /*implicit*/ ExpandingArray(const std::array<T, D>& values)
      : values_(values) {}

  /// Accesses the underlying `std::array`.
  std::array<T, D>& operator*() {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>& operator*() const {
    return values_;
  }

  /// Accesses the underlying `std::array`.
  std::array<T, D>* operator->() {
    return &values_;
  }

  /// Accesses the underlying `std::array`.
  const std::array<T, D>* operator->() const {
    return &values_;
  }

  /// Returns an `ArrayRef` to the underlying `std::array`.
  operator at::ArrayRef<T>() const {
    return values_;
  }

  /// Returns the extent of the `ExpandingArray`.
  size_t size() const noexcept {
    return D;
  }

 private:
  /// The backing array.
  std::array<T, D> values_;
};

} // namespace torch
