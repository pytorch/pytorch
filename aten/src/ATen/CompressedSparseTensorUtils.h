#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/CompressedSparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace sparse {
  inline CompressedSparseTensorImpl* get_sparse_csr_impl(const SparseTensor& self) {
    AT_ASSERTM(self.is_sparse_csr(), 
      "_internal_get_CompressedSparseTensorImpl: not a sparse GCS tensor");
    return static_cast<CompressedSparseTensorImpl*>(self.unsafeGetTensorImpl());
  }
} } // namespace at::sparse