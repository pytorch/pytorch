#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace sparse_csr {

using SparseCsrTensor = Tensor;

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  switch (self.layout()) {
  case kSparseCsr:
  case kSparseCsc:
  case kSparseBsr:
  case kSparseBsc:
    break;
  default:
    AT_ASSERTM(
               false,
               "_internal_get_SparseCsrTensorImpl: not a sparse CSR|CSC|BSR|BSC tensor");
  }
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}
} // namespace sparse
} // namespace at
