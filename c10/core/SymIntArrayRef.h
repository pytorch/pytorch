#pragma once

#include <c10/core/SymInt.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <array>
#include <initializer_list>
#include <iterator>
#include <vector>

namespace c10 {
using SymIntArrayRef = ArrayRef<SymInt>;

template <>
class ArrayRef<SymInt> final {
  using T = SymInt;

 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

  enum Unchecked { UNCHECKED };

 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_type Length;

  /// A bool saying if any element is symbolic
  bool AnySymbolic;

  void debugCheckNullptrInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        Data != nullptr || Length == 0,
        "created ArrayRef with nullptr and non-zero length! c10::optional relies on this being illegal");
  }

  void debugCheckAnySymbolicInvariant() {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        AnySymbolic == computeSymbolic(),
        "created SymIntArrayRef with incorrect AnySymbolic tag; real tag is ",
        !AnySymbolic);
  }

  bool computeSymbolic() const {
    for (const auto& s : *this) {
      if (s.is_symbolic())
        return true;
    }
    return false;
  }

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty ArrayRef.
  /* implicit */ constexpr ArrayRef()
      : Data(nullptr), Length(0), AnySymbolic(false) {}

  /// Construct an ArrayRef from a single element.
  ArrayRef(const T& OneElt)
      : Data(&OneElt), Length(1), AnySymbolic(OneElt.is_symbolic()) {}

  /// Construct an ArrayRef from a pointer and length.
  ArrayRef(const T* data, size_t length)
      : Data(data), Length(length), AnySymbolic(computeSymbolic()) {
    debugCheckNullptrInvariant();
  }

  ArrayRef(Unchecked, const int64_t* data, size_t length)
      : Data(reinterpret_cast<const c10::SymInt*>(data)),
        Length(length),
        AnySymbolic(false) {
    debugCheckNullptrInvariant();
    debugCheckAnySymbolicInvariant();
  }

  /// Construct an ArrayRef from a range.
  ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin), AnySymbolic(computeSymbolic()) {
    debugCheckNullptrInvariant();
  }

  /// Construct an ArrayRef from a SmallVector. This is templated in order to
  /// avoid instantiating SmallVectorTemplateCommon<T> whenever we
  /// copy-construct an ArrayRef.
  template <typename U>
  /* implicit */ ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec)
      : Data(Vec.data()), Length(Vec.size()), AnySymbolic(computeSymbolic()) {
    debugCheckNullptrInvariant();
  }

  template <
      typename Container,
      typename = std::enable_if_t<std::is_same<
          std::remove_const_t<decltype(std::declval<Container>().data())>,
          T*>::value>>
  /* implicit */ ArrayRef(const Container& container)
      : Data(container.data()),
        Length(container.size()),
        AnySymbolic(computeSymbolic()) {
    debugCheckNullptrInvariant();
  }

  /// Construct an ArrayRef from a std::vector.
  // The enable_if stuff here makes sure that this isn't used for
  // std::vector<bool>, because ArrayRef can't work on a std::vector<bool>
  // bitfield.
  template <typename A>
  /* implicit */ ArrayRef(const std::vector<T, A>& Vec)
      : Data(Vec.data()), Length(Vec.size()), AnySymbolic(computeSymbolic()) {
    static_assert(
        !std::is_same<T, bool>::value,
        "ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
  }

  /// Construct an ArrayRef from a std::array
  template <size_t N>
  /* implicit */ ArrayRef(const std::array<T, N>& Arr)
      : Data(Arr.data()), Length(N), AnySymbolic(computeSymbolic()) {}

  /// Construct an ArrayRef from a C array.
  template <size_t N>
  /* implicit */ ArrayRef(const T (&Arr)[N])
      : Data(Arr), Length(N), AnySymbolic(computeSymbolic()) {}

  /// Construct an ArrayRef from a std::initializer_list.
  /* implicit */ ArrayRef(const std::initializer_list<T>& Vec)
      : Data(
            std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                             : std::begin(Vec)),
        Length(Vec.size()),
        AnySymbolic(computeSymbolic()) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return Data;
  }
  constexpr iterator end() const {
    return Data + Length;
  }

  // These are actually the same as iterator, since ArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return Data;
  }
  constexpr const_iterator cend() const {
    return Data + Length;
  }

  reverse_iterator rbegin() const {
    return reverse_iterator(end());
  }
  reverse_iterator rend() const {
    return reverse_iterator(begin());
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return Length == 0;
  }

  constexpr const T* data() const {
    return Data;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return Length;
  }

  /// front - Get the first element.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& front() const {
    TORCH_CHECK(
        !empty(), "ArrayRef: attempted to access front() of empty list");
    return Data[0];
  }

  /// back - Get the last element.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& back() const {
    TORCH_CHECK(!empty(), "ArrayRef: attempted to access back() of empty list");
    return Data[Length - 1];
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  ArrayRef<T> slice(size_t N, size_t M) const {
    TORCH_CHECK(
        N + M <= size(),
        "ArrayRef: invalid slice, N = ",
        N,
        "; M = ",
        M,
        "; size = ",
        size());
    if (AnySymbolic) {
      // Recompute the field
      return ArrayRef<T>(data() + N, M);
    } else {
      // Definitely false
      return ArrayRef<T>(
          UNCHECKED, reinterpret_cast<const int64_t*>(data() + N), M);
    }
  }

  /// slice(n) - Chop off the first N elements of the array.
  ArrayRef<T> slice(size_t N) const {
    return slice(N, size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Vector compatibility
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const T& at(size_t Index) const {
    TORCH_CHECK(
        Index < Length,
        "ArrayRef: invalid index Index = ",
        Index,
        "; Length = ",
        Length);
    return Data[Index];
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, T>::value, ArrayRef<T>>::type&
  operator=(std::initializer_list<U>) = delete;

  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA bool any_symbolic() const {
    return AnySymbolic;
  }

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<T> vec() const {
    return std::vector<T>(Data, Data + Length);
  }

  /// @}
};

template <>
class arrayref_optional_base<ArrayRef<SymInt>> {
  using ArrayRefT = ArrayRef<SymInt>;

 public:
  union storage {
    struct raw {
      // ArrayRef has the invariant that if Data is nullptr then
      // Length must be zero, so this is an unused bit pattern.
      const void* p = nullptr;
      size_t sz = 1;
      bool b = false;
    } uninitialized_{};
    ArrayRefT value_;

    constexpr storage() noexcept : uninitialized_() {
      setUninitialized();
    }

    constexpr void setUninitialized() noexcept {
      uninitialized_.p = nullptr;
      uninitialized_.sz = 1;
      uninitialized_.b = false;
    }

    explicit constexpr storage(ArrayRefT& v) : value_(v) {}

    template <typename T>
    explicit constexpr storage(const std::initializer_list<T>& v) : value_(v) {}

    template <class... Args>
    explicit constexpr storage(Args&&... args)
        : value_(constexpr_forward<Args>(args)...) {}
  };

  storage storage_;

  constexpr arrayref_optional_base() noexcept = default;

  explicit constexpr arrayref_optional_base(const ArrayRefT& v) : storage_(v) {}

  template <class... Args>
  explicit constexpr arrayref_optional_base(in_place_t, Args&&... args)
      : storage_(constexpr_forward<Args>(args)...) {}

  template <typename T>
  explicit constexpr arrayref_optional_base(
      in_place_t,
      const std::initializer_list<T>& v)
      : storage_(v) {}

  constexpr bool initialized() const noexcept {
    typename storage::raw repr;
    // Cast to void* to suppress GCC's -Wclass-memaccess.
    memcpy(
        static_cast<void*>(&repr),
        static_cast<const void*>(&storage_),
        sizeof(storage_));
    return repr.p != nullptr || repr.sz == 0;
  }

  void setInitialized(bool init) noexcept {
    if (!init) {
      storage_.setUninitialized();
    } else {
      assert(initialized());
    }
  }
};

// NB: this is actually not that slow anymore
TORCH_API at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar);
TORCH_API at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar);
TORCH_API c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar);

// Prefer using a more semantic constructor, like
// fromIntArrayRefKnownNonNegative
inline SymIntArrayRef fromIntArrayRefUnchecked(IntArrayRef array_ref) {
  return SymIntArrayRef(
      SymIntArrayRef::UNCHECKED, array_ref.data(), array_ref.size());
}

inline SymIntArrayRef fromIntArrayRefKnownNonNegative(IntArrayRef array_ref) {
  return fromIntArrayRefUnchecked(array_ref);
}

inline SymIntArrayRef fromIntArrayRef(IntArrayRef array_ref) {
  for (size_t i = 0; i < array_ref.size(); ++i) {
    TORCH_CHECK(
        SymInt::check_range(array_ref[i]),
        "IntArrayRef contains an int that cannot be represented as a SymInt: ",
        array_ref[i]);
  }
  return fromIntArrayRefUnchecked(array_ref);
}

} // namespace c10
