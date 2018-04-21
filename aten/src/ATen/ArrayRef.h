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

#include <ATen/Error.h>
#include <ATen/SmallVector.h>

#include <array>
#include <iterator>
#include <vector>

namespace at {
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
  template<typename T>
  class ArrayRef {
  public:
    typedef const T *iterator;
    typedef const T *const_iterator;
    typedef size_t size_type;

    typedef std::reverse_iterator<iterator> reverse_iterator;

  private:
    /// The start of the array, in an external buffer.
    const T *Data;

    /// The number of elements.
    size_type Length;

  public:
    /// @name Constructors
    /// @{

    /// Construct an empty ArrayRef.
    /*implicit*/ ArrayRef() : Data(nullptr), Length(0) {}

    /// Construct an ArrayRef from a single element.
    /*implicit*/ ArrayRef(const T &OneElt)
      : Data(&OneElt), Length(1) {}

    /// Construct an ArrayRef from a pointer and length.
    /*implicit*/ ArrayRef(const T *data, size_t length)
      : Data(data), Length(length) {}

    /// Construct an ArrayRef from a range.
    ArrayRef(const T *begin, const T *end)
      : Data(begin), Length(end - begin) {}

    /// Construct an ArrayRef from a SmallVector. This is templated in order to
    /// avoid instantiating SmallVectorTemplateCommon<T> whenever we
    /// copy-construct an ArrayRef.
    template<typename U>
    /*implicit*/ ArrayRef(const SmallVectorTemplateCommon<T, U> &Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    }

    /// Construct an ArrayRef from a std::vector.
    template<typename A>
    /*implicit*/ ArrayRef(const std::vector<T, A> &Vec)
      : Data(Vec.data()), Length(Vec.size()) {}

    /// Construct an ArrayRef from a std::array
    template <size_t N>
    /*implicit*/ constexpr ArrayRef(const std::array<T, N> &Arr)
        : Data(Arr.data()), Length(N) {}

    /// Construct an ArrayRef from a C array.
    template <size_t N>
    /*implicit*/ constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

    /// Construct an ArrayRef from a std::initializer_list.
    /*implicit*/ ArrayRef(const std::initializer_list<T> &Vec)
    : Data(Vec.begin() == Vec.end() ? (T*)nullptr : Vec.begin()),
      Length(Vec.size()) {}

    /// @}
    /// @name Simple Operations
    /// @{

    iterator begin() const { return Data; }
    iterator end() const { return Data + Length; }

    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }

    /// empty - Check if the array is empty.
    bool empty() const { return Length == 0; }

    const T *data() const { return Data; }

    /// size - Get the array size.
    size_t size() const { return Length; }

    /// front - Get the first element.
    const T &front() const {
      AT_ASSERT(!empty(), "Empty list!");
      return Data[0];
    }

    /// back - Get the last element.
    const T &back() const {
      AT_ASSERT(!empty(), "Empty list!");
      return Data[Length-1];
    }

    /// equals - Check for element-wise equality.
    bool equals(ArrayRef RHS) const {
      if (Length != RHS.Length)
        return false;
      return std::equal(begin(), end(), RHS.begin());
    }

    /// slice(n, m) - Chop off the first N elements of the array, and keep M
    /// elements in the array.
    ArrayRef<T> slice(size_t N, size_t M) const {
      AT_ASSERT(N+M <= size(), "Invalid specifier");
      return ArrayRef<T>(data()+N, M);
    }

    /// slice(n) - Chop off the first N elements of the array.
    ArrayRef<T> slice(size_t N) const { return slice(N, size() - N); }

    /// @}
    /// @name Operator Overloads
    /// @{
    const T &operator[](size_t Index) const {
      return Data[Index];
    }

    /// Vector compatibility
    const T &at(size_t Index) const {
      AT_ASSERT(Index < Length, "Invalid index!");
      return Data[Index];
    }

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
    operator=(U &&Temporary) = delete;

    /// Disallow accidental assignment from a temporary.
    ///
    /// The declaration here is extra complicated so that "arrayRef = {}"
    /// continues to select the move assignment operator.
    template <typename U>
    typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type &
    operator=(std::initializer_list<U>) = delete;

    /// @}
    /// @name Expensive Operations
    /// @{
    std::vector<T> vec() const {
      return std::vector<T>(Data, Data+Length);
    }

    /// @}
    /// @name Conversion operators
    /// @{
    operator std::vector<T>() const {
      return std::vector<T>(Data, Data+Length);
    }

    /// @}
  };

} // end namespace at
