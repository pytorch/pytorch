#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at {
namespace sparse_csr {

using SparseCsrTensor = Tensor;

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  AT_ASSERTM(
      self.is_sparse_csr(),
      "_internal_get_SparseCsrTensorImpl: not a sparse CSR tensor");
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}
} // namespace sparse
} // namespace at
