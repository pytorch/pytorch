//===--- ArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// ATen: modified from llvm::ArrayRef.
// removed llvm-specific functionality
// removed some implicit const -> non-const conversions that rely on
// complicated std::enable_if meta-programming
// removed a bunch of slice variants for simplicity...

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <vector>

namespace c10 {
/// ArrayRef - Represent a constant reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the ArrayRef. For this reason, it is not in general
/// safe to store an ArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
///
/// NOTE: We have refactored out the headeronly parts of the ArrayRef struct
/// into HeaderOnlyArrayRef. As adding `virtual` would change the performance of
/// the underlying constexpr calls, we rely on apparent-type dispatch for
/// inheritance. This should be fine because their memory format is the same,
/// and it is never incorrect for ArrayRef to call HeaderOnlyArrayRef methods.
/// However, you should prefer to use ArrayRef when possible, because its use
/// of TORCH_CHECK will lead to better user-facing error messages.
template <typename T>
// ArrayRef cannot be derived from. Normally, we would use `final`
// specifier to force this constraint at compile time.  However, Intel
// compiler does not recognize ArrayRef as a class template (which is
// required in the definition of at::TensorAccessor, for instance)
// when `final` specifier is used. So, we cannot define ArrayRef as
// final because of the Intel compiler issue.
class ArrayRef : public HeaderOnlyArrayRef<T> {
 public:
  /// @name Constructors, all inherited from HeaderOnlyArrayRef except for
  /// SmallVector. As inherited constructors won't work with class template
  /// argument deduction (CTAD) until C++23, we add deduction guides after
  /// the class definition to enable CTAD.
  /// @{

  using HeaderOnlyArrayRef<T>::HeaderOnlyArrayRef;

  /// Construct an ArrayRef from a SmallVector. This is templated in order to
  /// avoid instantiating SmallVectorTemplateCommon<T> whenever we
  /// copy-construct an ArrayRef.
  /// NOTE: this is the only constructor that is not inherited from
  /// HeaderOnlyArrayRef.
  template <typename U>
  /* implicit */ ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : HeaderOnlyArrayRef<T>(Vec.data(), Vec.size()) {}

  /// @}
  /// @name Simple Operations, mostly inherited from HeaderOnlyArrayRef
  /// @{

  /// front - Get the first element.
  /// We deviate from HeaderOnlyArrayRef by using TORCH_CHECK instead of
  /// STD_TORCH_CHECK
  constexpr const T& front() const {
    TORCH_CHECK(
        !this->empty(), "ArrayRef: attempted to access front() of empty list");
    return this->Data[0];
  }

  /// back - Get the last element.
  /// We deviate from HeaderOnlyArrayRef by using TORCH_CHECK instead of
  /// STD_TORCH_CHECK
  constexpr const T& back() const {
    TORCH_CHECK(
        !this->empty(), "ArrayRef: attempted to access back() of empty list");
    return this->Data[this->Length - 1];
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  /// We deviate from HeaderOnlyArrayRef by using TORCH_CHECK instead of
  /// STD_TORCH_CHECK
  constexpr ArrayRef<T> slice(size_t N, size_t M) const {
    TORCH_CHECK(
        N + M <= this->size(),
        "ArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        this->size());
    return ArrayRef<T>(this->data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  /// We deviate from HeaderOnlyArrayRef by using TORCH_CHECK instead of
  /// STD_TORCH_CHECK
  constexpr ArrayRef<T> slice(size_t N) const {
    TORCH_CHECK(
        N <= this->size(),
        "ArrayRef: invalid slice, N = ",
        N,
        "; size = ",
        this->size());
    return slice(N, this->size() - N); // should this slice be this->slice?
  }

  /// @}
  /// @name Operator Overloads
  /// @{

  /// Vector compatibility
  /// We deviate from HeaderOnlyArrayRef by using TORCH_CHECK instead of
  /// STD_TORCH_CHECK
  constexpr const T& at(size_t Index) const {
    TORCH_CHECK(
        Index < this->Length,
        "ArrayRef: invalid index Index = ",
        Index,
        "; Length = ",
        this->Length);
    return this->Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, ArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;

  /// @}
};

/// Deduction guides for ArrayRef to support CTAD with inherited constructors
/// These mirror the constructors inherited from HeaderOnlyArrayRef
/// @{

// Single element constructor
template <typename T>
ArrayRef(const T&) -> ArrayRef<T>;

// Pointer and length constructor
template <typename T>
ArrayRef(const T*, size_t) -> ArrayRef<T>;

// Range constructor (begin, end)
template <typename T>
ArrayRef(const T*, const T*) -> ArrayRef<T>;

// Generic container constructor (anything with .data() and .size())
template <typename Container>
ArrayRef(const Container&) -> ArrayRef<
    std::remove_pointer_t<decltype(std::declval<Container>().data())>>;

// std::vector constructor
template <typename T, typename A>
ArrayRef(const std::vector<T, A>&) -> ArrayRef<T>;

// std::array constructor
template <typename T, size_t N>
ArrayRef(const std::array<T, N>&) -> ArrayRef<T>;

// C array constructor
template <typename T, size_t N>
ArrayRef(const T (&)[N]) -> ArrayRef<T>;

// std::initializer_list constructor
template <typename T>
ArrayRef(const std::initializer_list<T>&) -> ArrayRef<T>;

/// @}

template <typename T>
std::ostream& operator<<(std::ostream& out, ArrayRef<T> list) {
  int i = 0;
  out << "[";
  for (const auto& e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

/// @name ArrayRef Convenience constructors
/// @{

/// Construct an ArrayRef from a single element.
template <typename T>
ArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

/// Construct an ArrayRef from a pointer and length.
template <typename T>
ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return ArrayRef<T>(data, length);
}

/// Construct an ArrayRef from a range.
template <typename T>
ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return ArrayRef<T>(begin, end);
}

/// Construct an ArrayRef from a SmallVector.
template <typename T>
ArrayRef<T> makeArrayRef(const SmallVectorImpl<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a SmallVector.
template <typename T, unsigned N>
ArrayRef<T> makeArrayRef(const SmallVector<T, N>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a std::vector.
template <typename T>
ArrayRef<T> makeArrayRef(const std::vector<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a std::array.
template <typename T, std::size_t N>
ArrayRef<T> makeArrayRef(const std::array<T, N>& Arr) {
  return Arr;
}

/// Construct an ArrayRef from an ArrayRef (no-op) (const)
template <typename T>
ArrayRef<T> makeArrayRef(const ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from an ArrayRef (no-op)
template <typename T>
ArrayRef<T>& makeArrayRef(ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a C array.
template <typename T, size_t N>
// NOLINTNEXTLINE(*c-arrays*)
ArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return ArrayRef<T>(Arr);
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an c10::ArrayRef, which is why we need so
// many overloads.

template <typename T>
bool operator==(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return !c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(c10::ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(c10::ArrayRef<T>(a2));
}

using IntArrayRef = ArrayRef<int64_t>;

using IntList [[deprecated(
    "This alias is deprecated because it doesn't make ownership semantics obvious. Use IntArrayRef instead!")]] =
    ArrayRef<int64_t>;

} // namespace c10
