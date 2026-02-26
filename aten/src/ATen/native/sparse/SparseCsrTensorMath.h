#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>

namespace at::native::sparse::impl {

// Returns true if all entries of self are zero
// TODO: This has potential to be a generic helper
inline bool _is_sparse_and_zero(const Tensor& self) {
  if (self.layout() == kSparse || self.layout() == kSparseCsr ||
      self.layout() == kSparseCsc || self.layout() == kSparseBsr ||
      self.layout() == kSparseBsc) {
    if (self._nnz() == 0) {
      return true;
    }
  }
  return false;
}

inline void _check_is_cpu(const Tensor& self, std::string_view name) {
  TORCH_CHECK(
      self.is_cpu(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be CPU tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_is_cuda(const Tensor& self, std::string_view name) {
  TORCH_CHECK(
      self.is_cuda(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be CUDA tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_dim(const Tensor& self, int64_t target_dim, std::string_view name) {
  if (target_dim == 2) {
    TORCH_CHECK(
        self.dim() == target_dim,
        name, " must be a matrix, ",
        "got ", self.dim(), "-D tensor");
  }
  TORCH_CHECK(
      self.dim() == target_dim,
      "Expected ",
      name,
      " to be of dimension ",
      target_dim,
      " but got ",
      self.dim(),
      " instead.");
}

template <bool train>
inline void check_sparse_mm_reduce_impl_inputs(
    const Tensor& self,
    const Tensor& grad_out,
    const Tensor& other) {
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());

  const auto input_scalar_type = self.values().scalar_type();
  CheckedFrom c = train ? "sparse_mm_reduce_backward" : "sparse_mm_reduce";
  if (train) {
    checkLayout(c, grad_out, kStrided);
    checkScalarType(c, {grad_out, "grad_out", 1}, input_scalar_type);
    check_dim_size(grad_out, 2, 0, self.size(0));
    check_dim_size(grad_out, 2, 1, other.size(1));
  }

  int pos = train ? 2 : 1;
  checkLayout(c, other, kStrided);
  checkScalarType(c, {other, "other", pos}, input_scalar_type);
  check_dim_size(other, 2, 0, self.size(1));
}

} // at::native::sparse::impl
