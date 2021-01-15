#pragma once

#include <ATen/ATen.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/CompresedSparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

namespace at { namespace sparse {
  inline SparseGCSTensorImpl* get_sparse_gcs_impl(const SparseTensor& self) {
    AT_ASSERTM(self.is_sparse_gcs(), 
      "_internal_get_SparseGCSTensorImpl: not a sparse GCS tensor");
    return static_cast<SparseGCSTensorImpl*>(self.unsafeGetTensorImpl());
  }
} } // namespace at::sparse