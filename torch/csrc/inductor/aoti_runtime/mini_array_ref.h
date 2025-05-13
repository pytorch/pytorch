#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace torch::aot_inductor {

// Can't use c10::ArrayRef because it's not truly header-only and
// pulls in other c10 headers. This is (sadly) copy-pasted and
// adapted.
template <typename T>
class MiniArrayRef final {
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

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty MiniArrayRef.
  /* implicit */ constexpr MiniArrayRef() : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a single element.
  // TODO Make this explicit
  constexpr MiniArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}

  /// Construct an MiniArrayRef from a pointer and length.
  constexpr MiniArrayRef(T* data, size_t length) : Data(data), Length(length) {}

  /// Construct an MiniArrayRef from a range.
  constexpr MiniArrayRef(T* begin, T* end) : Data(begin), Length(end - begin) {}

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same_v<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>>>
  /* implicit */ MiniArrayRef(Container& container)
      : Data(container.data()), Length(container.size()) {}

  /// Construct an MiniArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because MiniArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  /* implicit */ MiniArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "MiniArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an MiniArrayRef from a std::array
  template <size_t N>
  /* implicit */ constexpr MiniArrayRef(std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N) {}

  /// Construct an MiniArrayRef from a C array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-array*)
  /* implicit */ constexpr MiniArrayRef(T (&Arr)[N]) : Data(Arr), Length(N) {}

  // /// Construct an MiniArrayRef from an empty C array.
  /* implicit */ constexpr MiniArrayRef(const volatile void* Arr)
      : Data(nullptr), Length(0) {}

  /// Construct an MiniArrayRef from a std::initializer_list.
  /* implicit */ constexpr MiniArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return Data;
  }
  constexpr iterator end() const {
    return Data + Length;
  }

  // These are actually the same as iterator, since MiniArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return Data;
  }
  constexpr const_iterator cend() const {
    return Data + Length;
  }

  constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return Length == 0;
  }

  constexpr T* data() const {
    return Data;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return Length;
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(MiniArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MiniArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MiniArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;
};

} // namespace torch::aot_inductor
