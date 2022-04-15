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
               "_internal_get_SparseCsrTensorImpl: expected sparse compressed tensor layout but got ", self.layout());
  }
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}

} // namespace sparse_csr
} // namespace at
