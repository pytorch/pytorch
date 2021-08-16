//===--- MutableArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Modified from c10::ArrayRef and modeled after LLVM's MutableArrayRef

#pragma once

#include <c10/util/C++17.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#include <array>
#include <iterator>
#include <vector>

namespace c10 {

/// MutableArrayRef - Represent a reference to an array (0 or more elements
/// consecutively in memory), i.e. a start pointer and a length.  It allows
/// various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data. It is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the MutableArrayRef. For this reason, it is not in general
/// safe to store an MutableArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
template <typename T>
class MutableArrayRef final {
 public:
  using iterator = T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  /// The start of the array, in an external buffer.
  T* Data;

  /// The number of elements.
  size_type Length;

  void debugCheckNullptrInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        Data != nullptr || Length == 0,
        "created MutableArrayRef with nullptr and non-zero length! c10::optional relies on this being illegal");
  }

 public:
  /// Construct an empty MutableArrayRef
  /* implicit */ MutableArrayRef() : Data(nullptr), Length(0) {}

  /// Construct an MutableArrayRef from a single element
  MutableArrayRef(T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an MutableArrayRef from a pointer and length.
  /// CUDA 9.2 fails to compile of host-only function on device
  C10_HOST_CONSTEXPR_EXCEPT_CUDA92 MutableArrayRef(T* data, size_t length)
      : Data(data), Length(length) {
    debugCheckNullptrInvariant();
  }

  /// Construct an MutableArrayRef from a range.
  /// CUDA 9.2 fails to compile of host-only function on device
  C10_HOST_CONSTEXPR_EXCEPT_CUDA92 MutableArrayRef(T* begin, T* end)
      : Data(begin), Length(end - begin) {
    debugCheckNullptrInvariant();
  }

  /// Construct an MutableArrayRef from a SmallVector. This is templated in order to
  /// avoid instantiating SmallVectorTemplateCommon<T> whenever we
  /// copy-construct an MutableArrayRef
  template <typename U>
  /* implicit */ MutableArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    debugCheckNullptrInvariant();
  }

  /// Construct an MutableArrayRef from a generic Container
  template <
      typename Container,
      typename = std::enable_if_t<std::is_same<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>::value>>
  /* implicit */ MutableArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {
    debugCheckNullptrInvariant();
  }

  /// Construct an MutableArrayRef from a std::vector
  template <typename A>
  /* implicit */ MutableArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same<T, bool>::value,
        "MutableArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an MutableArrayRef from a std::array
  template <size_t N>
  /* implicit */ MutableArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an MutableArrayRef from a C array
  template <size_t N>
  /* implicit */ MutableArrayRef(T (&Arr)[N]) : Data(Arr), Length(N) {}

  /// Construct an MutableArrayRef from a std::initializer_list
  /* implicit */ MutableArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}

  iterator begin() const {
    return Data;
  }
  iterator end() const {
    return Data + Length;
  }

  const_iterator cbegin() const {
    return const_cast<T*>(Data);
  }
  const_iterator cend() const {
    return const_cast<T*>(Data + Length);
  }

  reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  bool empty() const {
    return Length == 0;
  }

  T* data() const {
    return Data;
  }

  size_t size() const {
    return Length;
  }

  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA T& front() {
    TORCH_CHECK(
        !empty(), "MutableArrayRef: attempted to access front() of empty list");
    return Data[0];
  }

  /// back - Get the last element.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA T& back() const {
    TORCH_CHECK(!empty(), "MutableArrayRef: attempted to access back() of empty list");
    return Data[Length - 1];
  }

  bool equals(MutableArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA MutableArrayRef<T> slice(size_t N, size_t M)
      const {
    TORCH_CHECK(
        N + M <= size(),
        "MutableArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        size());
    return MutableArrayRef<T>(data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  MutableArrayRef<T> slice(size_t N) const {
    return slice(N, size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Vector compatibility
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA T& at(size_t Index) const {
    TORCH_CHECK(
        Index < Length,
        "MutableArrayRef: invalid index Index = ",
        Index,
        "; Length = ",
        Length);
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "MutableArrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, MutableArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "MutableArrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, MutableArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const {
    return std::vector<T>(Data, Data + Length);
  }

  /// @}
};

template <typename T>
std::ostream& operator<<(std::ostream& out, MutableArrayRef<T> list) {
  int i = 0;
  out << "[";
  for (auto e : list) {
    if (i++ > 0)
      out << ", ";
    out << e;
  }
  out << "]";
  return out;
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an c10::MutableArrayRef, which is why we need so
// many overloads.

template <typename T>
bool operator==(c10::MutableArrayRef<T> a1, c10::MutableArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(c10::MutableArrayRef<T> a1, c10::MutableArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, c10::MutableArrayRef<T> a2) {
  return c10::MutableArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, c10::MutableArrayRef<T> a2) {
  return !c10::MutableArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(c10::MutableArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(c10::MutableArrayRef<T>(a2));
}

template <typename T>
bool operator!=(c10::MutableArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(c10::MutableArrayRef<T>(a2));
}

} // namespace c10
