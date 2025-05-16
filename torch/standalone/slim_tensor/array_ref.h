#pragma once

#include <torch/standalone/slim_tensor/shared_ptr.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace torch::standalone {

template <typename T>
class MaybeOwningArrayRef final {
 public:
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  /// The start of the array, in an external buffer.
  T* data_;

  /// The number of elements.
  size_t length_;

  using BaseT = std::remove_const_t<T>;
  SharedPtr<BaseT> owning_data_;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty MaybeOwningArrayRef.
  /* implicit */ constexpr MaybeOwningArrayRef() : data_(nullptr), length_(0) {}

  /// Construct an MaybeOwningArrayRef from a single element.
  // TODO Make this explicit
  constexpr MaybeOwningArrayRef(const T& OneElt) : data_(&OneElt), length_(1) {}

  /// Construct an MaybeOwningArrayRef from a pointer and length.
  constexpr MaybeOwningArrayRef(T* data, size_t length, bool owning = false)
      : data_(data), length_(length) {
    if (owning) {
      owning_data_ = SharedPtr<BaseT>(new BaseT[length_]);
      std::memcpy(owning_data_.get(), data, length_ * sizeof(T));
      data_ = owning_data_.get();
    }
  }

  MaybeOwningArrayRef(MaybeOwningArrayRef&& other) = default;
  MaybeOwningArrayRef(const MaybeOwningArrayRef& other) = default;
  MaybeOwningArrayRef& operator=(const MaybeOwningArrayRef& other) = default;
  MaybeOwningArrayRef& operator=(MaybeOwningArrayRef&& other) = default;

  ~MaybeOwningArrayRef() {
    if (owning_data_) {
      owning_data_.reset();
    }
  }

  /// Construct an MaybeOwningArrayRef from a range.
  constexpr MaybeOwningArrayRef(T* begin, T* end)
      : data_(begin), length_(end - begin) {}

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same_v<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>>>
  /* implicit */ MaybeOwningArrayRef(Container& container)
      : data_(container.data()), length_(container.size()) {}

  /// Construct an MaybeOwningArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because MaybeOwningArrayRef can't work on a
  // std::vector<bool> bitfield.
  template <typename A>
  /* implicit */ MaybeOwningArrayRef(const std::vector<T, A>& Vec)
      : data_(Vec.data()), length_(Vec.size()) {
    static_assert(
        !std::is_same_v<T, bool>,
        "MaybeOwningArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an MaybeOwningArrayRef from a std::array
  template <size_t N>
  /* implicit */ constexpr MaybeOwningArrayRef(std::array<T, N>& Arr)
      : data_(Arr.data()), length_(N) {}

  /// Construct an MaybeOwningArrayRef from a C array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-array*)
  /* implicit */ constexpr MaybeOwningArrayRef(T (&Arr)[N])
      : data_(Arr), length_(N) {}

  // /// Construct an MaybeOwningArrayRef from an empty C array.
  /* implicit */ constexpr MaybeOwningArrayRef(const volatile void* Arr)
      : data_(nullptr), length_(0) {}

  /// Construct an MaybeOwningArrayRef from a std::initializer_list.
  /* implicit */ constexpr MaybeOwningArrayRef(
      const std::initializer_list<T>& Vec)
      : data_(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        length_(Vec.size()) {}

  /// @}
  /// @name Simple Operations
  /// @{
  constexpr iterator begin() const {
    return data_;
  }
  constexpr iterator end() const {
    return data_ + length_;
  }

  // These are actually the same as iterator, since MaybeOwningArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return data_;
  }
  constexpr const_iterator cend() const {
    return data_ + length_;
  }

  constexpr reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return length_ == 0;
  }

  constexpr T* data() const {
    return data_;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return length_;
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(MaybeOwningArrayRef RHS) const {
    return length_ == RHS.length_ && std::equal(begin(), end(), RHS.begin());
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return data_[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MaybeOwningArrayRef<T>>& operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MaybeOwningArrayRef<T>>& operator=(
      std::initializer_list<U>) = delete;
};

using ArrayRef = MaybeOwningArrayRef<const int64_t>;

} // namespace torch::standalone
