#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

namespace c10 {

/// HOArrayRef (HO = HeaderOnly) - A subset of ArrayRef that is implemented only
/// in headers. This will be a base class from which ArrayRef inherits, so that
/// we can keep much of the implementation shared.
///
/// [HOArrayRef vs ArrayRef note]
/// As HOArrayRef is a subset of ArrayRef, it has slightly less functionality than
/// ArrayRef. We document the differences below:
/// 1. ArrayRef has a debug-only internal assert that prevents construction with a
//     nullptr Data and nonzero length in the following constructors: from a pointer
//     and length, from a range, from a templated Container with .data() and .size().
//     HOArrayRef does not make that check for any constructor.
/// 2. ArrayRef can be constructed from a SmallVector. HOArrayRef cannot.
/// 3. ArrayRef uses TORCH_CHECK. HOArrayRef uses headeronly STD_TORCH_CHECK,
///    which will output a std::runtime_error vs a c10::Error.
/// In all other aspects, HOArrayRef is identical to ArrayRef, with the positive
/// benefit of being header-only and thus independent of libtorch.so.
template <typename T>
class HOArrayRef {
 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

 protected:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_type Length;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty HOArrayRef.
  /* implicit */ constexpr HOArrayRef() : Data(nullptr), Length(0) {}

  /// Construct a HOArrayRef from a single element.
  // TODO Make this explicit
  constexpr HOArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct a HOArrayRef from a pointer and length.
  constexpr HOArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {
  }

  /// Construct a HOArrayRef from a range.
  constexpr HOArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {
  }

  template <
      typename Container,
      typename U = decltype(std::declval<Container>().data()),
      typename = std::enable_if_t<
          (std::is_same_v<U, T*> || std::is_same_v<U, T const*>)>>
  /* implicit */ HOArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {
  }

  /// Construct a HOArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because ArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  /* implicit */ HOArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "HOArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct a HOArrayRef from a std::array
  template <size_t N>
  /* implicit */ constexpr HOArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct a HOArrayRef from a C array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-arrays*)
  /* implicit */ constexpr HOArrayRef(const T (&Arr)[N])
      : Data(Arr), Length(N) {}

  /// Construct a HOArrayRef from a std::initializer_list.
  /* implicit */ constexpr HOArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return this->Data;
  }
  constexpr iterator end() const {
    return this->Data + this->Length;
  }

  // These are actually the same as iterator, since ArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return this->Data;
  }
  constexpr const_iterator cend() const {
    return this->Data + this->Length;
  }

  constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// Check if all elements in the array satisfy the given expression
  constexpr bool allMatch(const std::function<bool(const T&)>& pred) const {
    return std::all_of(cbegin(), cend(), pred);
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return this->Length == 0;
  }

  constexpr const T* data() const {
    return this->Data;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return this->Length;
  }

  /// front - Get the first element.
  constexpr const T& front() const {
    STD_TORCH_CHECK(
        !this->empty(),
        "HOArrayRef: attempted to access front() of empty list");
    return this->Data[0];
  }

  /// back - Get the last element.
  constexpr const T& back() const {
    STD_TORCH_CHECK(
        !this->empty(), "HOArrayRef: attempted to access back() of empty list");
    return this->Data[this->Length - 1];
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(HOArrayRef RHS) const {
    return this->Length == RHS.Length &&
        std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  constexpr HOArrayRef<T> slice(size_t N, size_t M) const {
    STD_TORCH_CHECK(
        N + M <= this->size(),
        "HOArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        this->size());
    return HOArrayRef<T>(this->data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  constexpr HOArrayRef<T> slice(size_t N) const {
    STD_TORCH_CHECK(
        N <= this->size(),
        "HOArrayRef: invalid slice, N = ",
        N,
        "; size = ",
        this->size());
    return slice(N, this->size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return this->Data[Index];
  }

  /// Vector compatibility
  constexpr const T& at(size_t Index) const {
    STD_TORCH_CHECK(
        Index < this->Length,
        "HOArrayRef: invalid index Index = ",
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
  std::enable_if_t<std::is_same_v<U, T>, HOArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, HOArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const {
    return std::vector<T>(this->Data, this->Data + this->Length);
  }

  /// @}
};

} // namespace c10

namespace torch::headeronly {
using c10::HOArrayRef;
using IntHOArrayRef = HOArrayRef<int64_t>;
} // namespace torch::headeronly
