#pragma once
#include <ATen/Utils.h>
#include <c10/util/ArrayRef.h>

#include <vector>

namespace at {
/// MatrixRef - Like an ArrayRef, but with an extra recorded strides so that
/// we can easily view it as a multidimensional array.
///
/// Like ArrayRef, this class does not own the underlying data, it is expected
/// to be used in situations where the data resides in some other buffer.
///
/// This is intended to be trivially copyable, so it should be passed by
/// value.
///
/// For now, 2D only (so the copies are actually cheap, without having
/// to write a SmallVector class) and contiguous only (so we can
/// return non-strided ArrayRef on index).
///
/// P.S. dimension 0 indexes rows, dimension 1 indexes columns
template <typename T>
class MatrixRef {
 public:
  typedef size_t size_type;

 private:
  /// Underlying ArrayRef
  ArrayRef<T> arr;

  /// Stride of dim 0 (outer dimension)
  size_type stride0;

  // Stride of dim 1 is assumed to be 1

 public:
  /// Construct an empty Matrixref.
  /*implicit*/ MatrixRef() : arr(nullptr), stride0(0) {}

  /// Construct an MatrixRef from an ArrayRef and outer stride.
  /*implicit*/ MatrixRef(ArrayRef<T> arr, size_type stride0)
      : arr(arr), stride0(stride0) {
    TORCH_CHECK(
        arr.size() % stride0 == 0,
        "MatrixRef: ArrayRef size ",
        arr.size(),
        " not divisible by stride ",
        stride0)
  }

  /// @}
  /// @name Simple Operations
  /// @{

  /// empty - Check if the matrix is empty.
  bool empty() const {
    return arr.empty();
  }

  const T* data() const {
    return arr.data();
  }

  /// size - Get size a dimension
  size_t size(size_t dim) const {
    if (dim == 0) {
      return arr.size() / stride0;
    } else if (dim == 1) {
      return stride0;
    } else {
      TORCH_CHECK(
          0, "MatrixRef: out of bounds dimension ", dim, "; expected 0 or 1");
    }
  }

  size_t numel() const {
    return arr.size();
  }

  /// equals - Check for element-wise equality.
  bool equals(MatrixRef RHS) const {
    return stride0 == RHS.stride0 && arr.equals(RHS.arr);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  ArrayRef<T> operator[](size_t Index) const {
    return arr.slice(Index * stride0, stride0);
  }

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MatrixRef<T>>& operator=(
      U&& Temporary) = delete;

  /// Disallow accidental assignment from a temporary.
  ///
  /// The declaration here is extra complicated so that "arrayRef = {}"
  /// continues to select the move assignment operator.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, MatrixRef<T>>& operator=(
      std::initializer_list<U>) = delete;
};

} // end namespace at
