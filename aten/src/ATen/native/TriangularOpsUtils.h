#include <ATen/core/Tensor.h>
#include <ATen/native/LinearAlgebraUtils.h>

namespace at {
namespace native {

/*
 * Given batches of matrices with arbitrary batch dim,
 * computes the number of batches for Triu and Tril. This ignores stride 0 dimension
 */
static inline int64_t batchCountTrilTriu(const Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    if (batched_matrices.stride(i) != 0) {
      result *= batched_matrices.size(i);
    }
  }
  return result;
}

/* Checks a necessary property for the triu and tril implementations, hence the name.
 * Here batch contiguity is checked for tensors with greater than 4 dimensions.
 * Contiguous tensors and tensors with less than 3 dimensions pass this check
 */
static inline std::tuple<bool, Tensor> checkTrilTriuBatchContiguous(const Tensor& tensor, bool allow_zero_stride) {
  // Complete contiguity is the most desired property, which is why
  // we return true if the tensor is contiguous
  if (tensor.is_contiguous()) {
    auto default_strides_for_size = batched_matrix_contiguous_strides(tensor.sizes());
    if (tensor.strides() == default_strides_for_size) {
      return std::make_tuple(true, tensor);
    } else {
      return std::make_tuple(false, tensor.as_strided(tensor.sizes(), default_strides_for_size));
    }
  }

  int64_t dims = tensor.dim();

  // Tensors with dimension less than 4 are handled by default
  if (allow_zero_stride && dims <= 3) {
    return std::make_tuple(true, tensor);
  }

  int64_t expected_stride = tensor.size(-1) * tensor.size(-2);
  for (int64_t i = dims - 3; i >= 0; i--) {
    // Skip trivial dimension;
    if (allow_zero_stride && i == 0 && (tensor.stride(i) == 0 || tensor.size(i) == 1)) {
      continue;
    }
    if (expected_stride != tensor.stride(i)) {
      return std::make_tuple(false, tensor.contiguous());
    }
    expected_stride *= tensor.size(i);
  }
  return std::make_tuple(true, tensor);
}

}  // namespace native
}  // namespace at
