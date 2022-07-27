#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {

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

inline void _check_is_cpu(const Tensor& self, c10::string_view name) {
  TORCH_CHECK(
      self.is_cpu(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be CPU tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_is_cuda(const Tensor& self, c10::string_view name) {
  TORCH_CHECK(
      self.is_cuda(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be CUDA tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_dim(const Tensor& self, int64_t target_dim, c10::string_view name) {
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

}
}
}
}
