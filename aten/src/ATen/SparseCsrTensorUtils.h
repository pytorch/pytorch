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

inline std::string layoutToString(Layout layout, bool upper=false, bool lower=false) {
  switch (layout) {
  case kSparseCsr: return (upper ? "CSR" : (lower ? "csr" : "Csr"));
  case kSparseCsc: return (upper ? "CSC" : (lower ? "csc" : "Csc"));
  case kSparseBsr: return (upper ? "BSR" : (lower ? "bsr" : "Bsr"));
  case kSparseBsc: return (upper ? "BSC" : (lower ? "bsc" : "Bsc"));
  default:
    TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
    return "";
  }
}

inline bool isCompressedRow(Layout layout) {
  return (layout == kSparseCsr || layout == kSparseBsr);
}

inline bool isCompressedColumn(Layout layout) {
  return (layout == kSparseCsc || layout == kSparseBsc);
}

inline std::string compressedIndicesName(Layout layout) {
  switch (layout) {
  case kSparseCsr:
  case kSparseBsr:
    return "crow_indices";
  case kSparseCsc:
  case kSparseBsc:
    return "ccol_indices";
  default:
    TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
    return "";
  }
}

inline std::string plainIndicesName(Layout layout) {
  switch (layout) {
  case kSparseCsr:
  case kSparseBsr:
    return "col_indices";
  case kSparseCsc:
  case kSparseBsc:
    return "row_indices";
  default:
    TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
    return "";
  }
}

inline int rowDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 2 : 1);
}

inline int columnDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedColumn(layout) ? 2 : 1);
}

} // namespace sparse_csr
} // namespace at
