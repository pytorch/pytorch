#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>

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

SPMM_REDUCE_OP get_operator_enum(const c10::string_view reduce) {
  if (reduce == "sum") {
    return SPMM_SUM;
  } else if (reduce == "mean") {
    return SPMM_MEAN;
  } else if (reduce == "max") {
    return SPMM_MAX;
  } else if (reduce == "min") {
    return SPMM_MIN;
  } else {
    TORCH_CHECK(false, "spmm_reduce: reduce argument must be either sum, mean, max or min.");
  }
}

template <bool train>
void check_spmm_reduce_inputs(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& row_indices,
    const Tensor& ccol_indices,
    const Tensor& csr2csc) {
  TORCH_CHECK(input.is_sparse_csr(),"Expected input to be sparse CSR tensor.");

  const auto input_scalar_type = input.values().scalar_type();
  const auto index_scalar_type = input.col_indices().scalar_type();
  int64_t nnz = input._nnz();

  CheckedFrom c = train ? "spmm_reduce_backward" : "spmm_reduce";
  if (train) {
    checkLayout(c, grad_output, kStrided);
    checkScalarType(c, {grad_output, "grad_output", 1}, input_scalar_type);
    check_dim_size(grad_output, 2, 0, input.size(0));
    check_dim_size(grad_output, 2, 1, weight.size(1));
  }

  int pos = train ? 2 : 1;
  checkLayout(c, weight, kStrided);
  checkScalarType(c, {weight, "weight", pos}, input_scalar_type);
  check_dim_size(weight, 2, 0, input.size(1));

  if (row_indices.defined()) {
    checkLayout(c, row_indices, kStrided);
    checkScalarType(c, {row_indices, "row_indices", pos++}, index_scalar_type);
    check_dim_size(row_indices, 1, 0, nnz);
  }
  if (ccol_indices.defined()) {
    checkLayout(c, ccol_indices, kStrided);
    checkScalarType(c, {ccol_indices, "ccol_indices", pos++}, index_scalar_type);
    check_dim_size(ccol_indices, 1, 0, input.size(1) + 1);
  }
  if (csr2csc.defined()) {
    checkLayout(c, csr2csc, kStrided);
    checkScalarType(c, {csr2csc, "csr2csc", pos++}, index_scalar_type);
    check_dim_size(csr2csc, 1, 0, nnz);
  }
}

}
}
}
}
