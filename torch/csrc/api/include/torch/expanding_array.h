#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
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
      : ExpandingArray(at::ArrayRef<T>(list)) {}

  /// Constructs an `ExpandingArray` from an `std::vector`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(std::vector<T> vec)
      : ExpandingArray(at::ArrayRef<T>(vec)) {}

  /// Constructs an `ExpandingArray` from an `at::ArrayRef`. The extent of
  /// the length is checked against the `ExpandingArray`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArray(at::ArrayRef<T> values) {
    // clang-format off
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
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

 protected:
  /// The backing array.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::array<T, D> values_;
};

template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArray<D, T>& expanding_array) {
  if (expanding_array.size() == 1) {
    return stream << expanding_array->at(0);
  }
  return stream << static_cast<at::ArrayRef<T>>(expanding_array);
}

/// A utility class that accepts either a container of `D`-many `c10::optional<T>` values,
/// or a single `c10::optional<T>` value, which is internally repeated `D` times.
/// It has the additional ability to accept containers of the underlying type `T` and
/// convert them to a container of `c10::optional<T>`.
template <size_t D, typename T = int64_t>
class ExpandingArrayWithOptionalElem : public ExpandingArray<D, c10::optional<T>> {
 public:
  using ExpandingArray<D, c10::optional<T>>::ExpandingArray;

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `initializer_list` of the underlying type `T`.
  /// The extent of the length is checked against the `ExpandingArrayWithOptionalElem`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(std::initializer_list<T> list)
      : ExpandingArrayWithOptionalElem(at::ArrayRef<T>(list)) {}

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `std::vector` of the underlying type `T`.
  /// The extent of the length is checked against the `ExpandingArrayWithOptionalElem`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(std::vector<T> vec)
      : ExpandingArrayWithOptionalElem(at::ArrayRef<T>(vec)) {}

  /// Constructs an `ExpandingArrayWithOptionalElem` from an `at::ArrayRef` of the underlying type `T`.
  /// The extent of the length is checked against the `ExpandingArrayWithOptionalElem`'s extent parameter `D`
  /// at runtime.
  /*implicit*/ ExpandingArrayWithOptionalElem(at::ArrayRef<T> values) : ExpandingArray<D, c10::optional<T>>(0) {
    // clang-format off
    TORCH_CHECK(
        values.size() == D,
        "Expected ", D, " values, but instead got ", values.size());
    // clang-format on
    for (size_t i = 0; i < this->values_.size(); i++) {
      this->values_[i] = values[i];
    }
  }

  /// Constructs an `ExpandingArrayWithOptionalElem` from a single value of the underlying type `T`,
  /// which is repeated `D` times (where `D` is the extent parameter of the `ExpandingArrayWithOptionalElem`).
  /*implicit*/ ExpandingArrayWithOptionalElem(T single_size) : ExpandingArray<D, c10::optional<T>>(0) {
    for (size_t i = 0; i < this->values_.size(); i++) {
      this->values_[i] = single_size;
    }
  }

  /// Constructs an `ExpandingArrayWithOptionalElem` from a correctly sized `std::array` of the underlying type `T`.
  /*implicit*/ ExpandingArrayWithOptionalElem(const std::array<T, D>& values) : ExpandingArray<D, c10::optional<T>>(0) {
    for (size_t i = 0; i < this->values_.size(); i++) {
      this->values_[i] = values[i];
    }
  }
};

template <size_t D, typename T>
std::ostream& operator<<(
    std::ostream& stream,
    const ExpandingArrayWithOptionalElem<D, T>& expanding_array_with_opt_elem) {
  if (expanding_array_with_opt_elem.size() == 1) {
    const auto& elem = expanding_array_with_opt_elem->at(0);
    stream << (elem.has_value() ? c10::str(elem.value()) : "None");
  } else {
    std::vector<std::string> str_array;
    for (const auto& elem : *expanding_array_with_opt_elem) {
      str_array.emplace_back(elem.has_value() ? c10::str(elem.value()) : "None");
    }
    stream << at::ArrayRef<std::string>(str_array);
  }
  return stream;
}

} // namespace torch
