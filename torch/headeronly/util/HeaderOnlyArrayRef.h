#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

namespace c10 {

/// HeaderOnlyArrayRef - A subset of ArrayRef that is implemented only
/// in headers. This will be a base class from which ArrayRef inherits, so that
/// we can keep much of the implementation shared.
///
/// [HeaderOnlyArrayRef vs ArrayRef note]
/// As HeaderOnlyArrayRef is a subset of ArrayRef, it has slightly less
/// functionality than ArrayRef. We document the minor differences below:
/// 1. ArrayRef has an extra convenience constructor for SmallVector.
/// 2. ArrayRef uses TORCH_CHECK. HeaderOnlyArrayRef uses header-only
///    STD_TORCH_CHECK, which will output a std::runtime_error vs a
///    c10::Error. Consequently, you should use ArrayRef when possible
///    and HeaderOnlyArrayRef only when necessary to support headeronly code.
/// In all other aspects, HeaderOnlyArrayRef is identical to ArrayRef, with the
/// positive benefit of being header-only and thus independent of libtorch.so.
template <typename T>
class HeaderOnlyArrayRef {
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

  /// Construct an empty HeaderOnlyArrayRef.
  /* implicit */ constexpr HeaderOnlyArrayRef() : Data(nullptr), Length(0) {}

  /// Construct a HeaderOnlyArrayRef from a single element.
  // TODO Make this explicit
  constexpr HeaderOnlyArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct a HeaderOnlyArrayRef from a pointer and length.
  constexpr HeaderOnlyArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {}

  /// Construct a HeaderOnlyArrayRef from a range.
  constexpr HeaderOnlyArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {}

  template <
      typename Container,
      typename U = decltype(std::declval<Container>().data()),
      typename = std::enable_if_t<
          (std::is_same_v<U, T*> || std::is_same_v<U, T const*>)>>
  /* implicit */ HeaderOnlyArrayRef(const Container& container)
      : Data(container.data()), Length(container.size()) {}

  /// Construct a HeaderOnlyArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because ArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  /* implicit */ HeaderOnlyArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "HeaderOnlyArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct a HeaderOnlyArrayRef from a std::array
  template <size_t N>
  /* implicit */ constexpr HeaderOnlyArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct a HeaderOnlyArrayRef from a C array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-arrays*)
  /* implicit */ constexpr HeaderOnlyArrayRef(const T (&Arr)[N])
      : Data(Arr), Length(N) {}

  /// Construct a HeaderOnlyArrayRef from a std::initializer_list.
  /* implicit */ constexpr HeaderOnlyArrayRef(
      const std::initializer_list<T>& Vec)
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
        "HeaderOnlyArrayRef: attempted to access front() of empty list");
    return this->Data[0];
  }

  /// back - Get the last element.
  constexpr const T& back() const {
    STD_TORCH_CHECK(
        !this->empty(),
        "HeaderOnlyArrayRef: attempted to access back() of empty list");
    return this->Data[this->Length - 1];
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(HeaderOnlyArrayRef RHS) const {
    return this->Length == RHS.Length &&
        std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  constexpr HeaderOnlyArrayRef<T> slice(size_t N, size_t M) const {
    STD_TORCH_CHECK(
        N + M <= this->size(),
        "HeaderOnlyArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        this->size());
    return HeaderOnlyArrayRef<T>(this->data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  constexpr HeaderOnlyArrayRef<T> slice(size_t N) const {
    STD_TORCH_CHECK(
        N <= this->size(),
        "HeaderOnlyArrayRef: invalid slice, N = ",
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
        "HeaderOnlyArrayRef: invalid index Index = ",
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
  std::enable_if_t<std::is_same_v<U, T>, HeaderOnlyArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, HeaderOnlyArrayRef<T>>& operator=(
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
using c10::HeaderOnlyArrayRef;
using IntHeaderOnlyArrayRef = HeaderOnlyArrayRef<int64_t>;
} // namespace torch::headeronly
