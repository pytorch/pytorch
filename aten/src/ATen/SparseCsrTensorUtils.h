#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>

#define AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, ...)    \
  [&] {                                                                 \
    const auto& the_layout = LAYOUT;                                    \
    switch (the_layout) {                                               \
    case kSparseCsr:                                                    \
    case kSparseCsc:                                                    \
    case kSparseBsr:                                                    \
    case kSparseBsc:                                                    \
      return __VA_ARGS__();                                             \
    default:                                                            \
      AT_ERROR(#NAME, " expected sparse compressed tensor layout but got ", the_layout); \
    }                                                                   \
  } ()

#define AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, ROW_DIM_ACTION, COLUMN_DIM_ACTION) \
  [&]() {                                                               \
    const auto& the_layout = LAYOUT;                                    \
    switch (the_layout) {                                               \
    case kSparseCsr:                                                    \
    case kSparseBsr:                                                    \
      return (ROW_DIM_ACTION)();                                        \
    case kSparseCsc:                                                    \
    case kSparseBsc:                                                    \
      return (COLUMN_DIM_ACTION)();                                     \
    default:                                                            \
      AT_ERROR(#NAME, " expected sparse compressed tensor layout but got ", the_layout); \
    }                                                                   \
  } ()

#define AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, NO_BLOCK_ACTION, BLOCK_ACTION) \
  [&]() {                                                               \
    const auto& the_layout = LAYOUT;                                    \
    switch (the_layout) {                                               \
    case kSparseCsr:                                                    \
    case kSparseCsc:                                                    \
      return (NO_BLOCK_ACTION)();                                       \
    case kSparseBsr:                                                    \
    case kSparseBsc:                                                    \
      return (BLOCK_ACTION)();                                          \
    default:                                                            \
      AT_ERROR(#NAME, " expected sparse compressed tensor layout but got ", the_layout); \
    }                                                                   \
  } ()

namespace at {
namespace sparse_csr {

using SparseCsrTensor = Tensor;

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(self.layout(), "get_sparse_csr_impl", [&] {});
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
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "isCompressedRow", [&]{ return true; }, [&]{ return false; });
}

inline bool isCompressedColumn(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "isCompressedColumn", [&]{ return false; }, [&]{ return true; });
}

inline std::string compressedIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "compressedIndicesName", [&]{ return "crow_indices"; }, [&]{ return "ccol_indices"; });
}

inline std::string plainIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout, "plainIndicesName", [&]{ return "col_indices"; }, [&]{ return "row_indices"; });
}

inline int rowDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 2 : 1);
}

inline int columnDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedColumn(layout) ? 2 : 1);
}

inline int compressedDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 2 : 1);
}

inline int plainDimension(Layout layout, IntArrayRef size) {
  return size.size() - (isCompressedRow(layout) ? 1 : 2);
}

} // namespace sparse_csr
} // namespace at
