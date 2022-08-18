// This file defines `SymIntArrayRef` which serves as the view onto
// std::vector<SymInt>. This class is conceptually and mostly functionally
// equivalent to ArrayRef<SymInt>.
//
// However, ArrayRef<SymInt> can't be used directly as it introduces ambiguity
// in the following cases:
//   - a.expand({1, 2, 3}) matches two overloads:
//       1. `at::Tensor Tensor::expand(c10::SymIntArrayRef size, bool implicit)`
//       2. `at::Tensor Tensor::expand(at::IntArrayRef size, bool implicit)`
// Introducing `SymIntArrayRef` allows to have a finer-grained control over
// which overload will be used.

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
/// SymIntArrayRef - Represent a constant reference to an array (0 or more
/// elements consecutively in memory), i.e. a start pointer and a length.  It
/// allows various APIs to take consecutive elements easily and conveniently.
///
/// This class does not own the underlying data, it is expected to be used in
/// situations where the data resides in some other buffer, whose lifetime
/// extends past that of the SymIntArrayRef. For this reason, it is not in
/// general safe to store an SymIntArrayRef.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.

class SymIntArrayRef final {
 public:
  using iterator = const c10::SymInt*;
  using const_iterator = const c10::SymInt*;
  using size_type = size_t;
  using value_type = c10::SymInt;

  using reverse_iterator = std::reverse_iterator<iterator>;

 private:
  ArrayRef<c10::SymInt> wrapped_symint_array_ref;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty SymIntArrayRef.
  /* implicit */ constexpr SymIntArrayRef() {}

  /* implicit */ SymIntArrayRef(const std::vector<c10::SymInt>& Vec)
      : wrapped_symint_array_ref(Vec) {}

  /// Construct an SymIntArrayRef from a pointer and length.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA SymIntArrayRef(
      const c10::SymInt* data,
      size_t length)
      : wrapped_symint_array_ref(data, length) {}

  template <typename U>
  /* implicit */ SymIntArrayRef(
      const SmallVectorTemplateCommon<c10::SymInt, U>& Vec)
      : wrapped_symint_array_ref(Vec) {}

  /// Construct an SymIntArrayRef from a range.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA SymIntArrayRef(
      const c10::SymInt* begin,
      const c10::SymInt* end)
      : wrapped_symint_array_ref(begin, end) {}

  /// Construct an SymIntArrayRef from a C array.
  template <size_t N>
  /* implicit */ constexpr SymIntArrayRef(const c10::SymInt (&Arr)[N])
      : wrapped_symint_array_ref(Arr) {}

  static SymIntArrayRef fromIntArrayRef(IntArrayRef array_ref) {
    for (size_t i = 0; i < array_ref.size(); ++i) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          SymInt::check_range(array_ref[i]),
          "IntArrayRef contains int that cannot be representative as a SymInt",
          array_ref[i]);
    }
    return SymIntArrayRef(
        reinterpret_cast<const SymInt*>(array_ref.data()), array_ref.size());
  }

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return wrapped_symint_array_ref.begin();
  }
  constexpr iterator end() const {
    return wrapped_symint_array_ref.end();
  }

  // These are actually the same as iterator, since SymIntArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return wrapped_symint_array_ref.cbegin();
  }
  constexpr const_iterator cend() const {
    return wrapped_symint_array_ref.cend();
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return size() == 0;
  }

  constexpr const c10::SymInt* data() const {
    return wrapped_symint_array_ref.data();
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return wrapped_symint_array_ref.size();
  }

  /// front - Get the first element.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const c10::SymInt& front() const {
    return wrapped_symint_array_ref.front();
  }

  /// back - Get the last element.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const c10::SymInt& back() const {
    return wrapped_symint_array_ref.back();
  }

  /// equals - Check for element-wise equality.
  constexpr bool equals(SymIntArrayRef RHS) const {
    return this->wrapped_symint_array_ref.equals(RHS.wrapped_symint_array_ref);
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA SymIntArrayRef
  slice(size_t N, size_t M) const {
    return SymIntArrayRef(wrapped_symint_array_ref.data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA SymIntArrayRef slice(size_t N) const {
    return slice(N, size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const c10::SymInt& operator[](size_t Index) const {
    return wrapped_symint_array_ref[Index];
  }

  /// Vector compatibility
  C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA const c10::SymInt& at(size_t Index) const {
    return wrapped_symint_array_ref.at(Index);
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, c10::SymInt>::value, SymIntArrayRef>::
      type&
      operator=(U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  typename std::enable_if<std::is_same<U, c10::SymInt>::value, SymIntArrayRef>::
      type&
      operator=(std::initializer_list<U>) = delete;

  /// @}
  /// @name Expensive Operations
  /// @{
  std::vector<c10::SymInt> vec() const {
    return wrapped_symint_array_ref.vec();
  }

  friend std::ostream& operator<<(
      std::ostream& out,
      const SymIntArrayRef& list);
  /// @}
};

TORCH_API at::IntArrayRef asIntArrayRefSlow(c10::SymIntArrayRef ar);
TORCH_API at::IntArrayRef asIntArrayRefUnchecked(c10::SymIntArrayRef ar);
TORCH_API c10::optional<at::IntArrayRef> asIntArrayRefSlowOpt(
    c10::SymIntArrayRef ar);

inline std::ostream& operator<<(
    std::ostream& out,
    const c10::SymIntArrayRef& list) {
  return out << list.wrapped_symint_array_ref;
}

} // namespace c10
