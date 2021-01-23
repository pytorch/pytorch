#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/CompressedRowSparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace sparse {
  inline CompressedRowSparseTensorImpl* get_sparse_csr_impl(const SparseTensor& self) {
    AT_ASSERTM(self.is_sparse_csr(), 
      "_internal_get_CompressedRowSparseTensorImpl: not a sparse CSR tensor");
    return static_cast<CompressedRowSparseTensorImpl*>(self.unsafeGetTensorImpl());
  }
} } // namespace at::sparse