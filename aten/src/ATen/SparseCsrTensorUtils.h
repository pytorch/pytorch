#pragma once

#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/core/Tensor.h>

#define AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(LAYOUT, NAME, ...) \
  [&] {                                                              \
    const auto& the_layout = LAYOUT;                                 \
    switch (the_layout) {                                            \
      case kSparseCsr:                                               \
      case kSparseCsc:                                               \
      case kSparseBsr:                                               \
      case kSparseBsc:                                               \
        return __VA_ARGS__();                                        \
      default:                                                       \
        AT_ERROR(                                                    \
            #NAME,                                                   \
            " expected sparse compressed tensor layout but got ",    \
            the_layout);                                             \
    }                                                                \
  }()

#define AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(                \
    LAYOUT, NAME, ROW_DIM_ACTION, COLUMN_DIM_ACTION)              \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      case kSparseCsr:                                            \
      case kSparseBsr:                                            \
        return (ROW_DIM_ACTION)();                                \
      case kSparseCsc:                                            \
      case kSparseBsc:                                            \
        return (COLUMN_DIM_ACTION)();                             \
      default:                                                    \
        AT_ERROR(                                                 \
            #NAME,                                                \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()

#define AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(              \
    LAYOUT, NAME, NO_BLOCK_ACTION, BLOCK_ACTION)                  \
  [&]() {                                                         \
    const auto& the_layout = LAYOUT;                              \
    switch (the_layout) {                                         \
      case kSparseCsr:                                            \
      case kSparseCsc:                                            \
        return (NO_BLOCK_ACTION)();                               \
      case kSparseBsr:                                            \
      case kSparseBsc:                                            \
        return (BLOCK_ACTION)();                                  \
      default:                                                    \
        AT_ERROR(                                                 \
            #NAME,                                                \
            " expected sparse compressed tensor layout but got ", \
            the_layout);                                          \
    }                                                             \
  }()

#define AT_DISPATCH_SPARSE_ROW_COMPRESSED_LAYOUTS(                    \
    LAYOUT, NAME, ROW_DIM_ACTION)                                     \
  [&]() {                                                             \
    const auto& the_layout = LAYOUT;                                  \
    switch (the_layout) {                                             \
      case kSparseCsr:                                                \
      case kSparseBsr:                                                \
        return (ROW_DIM_ACTION)();                                    \
      default:                                                        \
        AT_ERROR(                                                     \
            #NAME,                                                    \
            " expected sparse row compressed tensor layout but got ", \
            the_layout);                                              \
    }                                                                 \
  }()

#define AT_DISPATCH_SPARSE_COL_COMPRESSED_LAYOUTS(                       \
    LAYOUT, NAME, COL_DIM_ACTION)                                        \
  [&]() {                                                                \
    const auto& the_layout = LAYOUT;                                     \
    switch (the_layout) {                                                \
      case kSparseCsc:                                                   \
      case kSparseBsc:                                                   \
        return (COL_DIM_ACTION)();                                       \
      default:                                                           \
        AT_ERROR(                                                        \
            #NAME,                                                       \
            " expected sparse column compressed tensor layout but got ", \
            the_layout);                                                 \
    }                                                                    \
  }()

#define AT_DISPATCH_SPARSE_COMPRESSED_NONBLOCK_LAYOUTS(LAYOUT, NAME, ACTION)  \
  [&]() {                                                                     \
    const auto& the_layout = LAYOUT;                                          \
    switch (the_layout) {                                                     \
      case kSparseCsr:                                                        \
      case kSparseCsc:                                                        \
        return (ACTION)();                                                    \
      default:                                                                \
        AT_ERROR(                                                             \
            #NAME,                                                            \
            " expected sparse compressed (non-block) tensor layout but got ", \
            the_layout);                                                      \
    }                                                                         \
  }()

#define AT_DISPATCH_SPARSE_COMPRESSED_BLOCK_LAYOUTS(LAYOUT, NAME, ACTION) \
  [&]() {                                                                 \
    const auto& the_layout = LAYOUT;                                      \
    switch (the_layout) {                                                 \
      case kSparseBsr:                                                    \
      case kSparseBsc:                                                    \
        return (ACTION)();                                                \
      default:                                                            \
        AT_ERROR(                                                         \
            #NAME,                                                        \
            " expected sparse compressed block tensor layout but got ",   \
            the_layout);                                                  \
    }                                                                     \
  }()

namespace at {
namespace sparse_csr {

using SparseCsrTensor = Tensor;

inline bool is_sparse_compressed(const Layout& layout) {
  switch (layout) {
    case kSparseCsr:
    case kSparseCsc:
    case kSparseBsr:
    case kSparseBsc:
      return true;
    default:;
  }
  return false;
}

inline bool is_sparse_compressed(const Tensor& self) {
  return is_sparse_compressed(self.layout());
}

inline SparseCsrTensorImpl* get_sparse_csr_impl(const SparseCsrTensor& self) {
  AT_DISPATCH_ALL_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(), "get_sparse_csr_impl", [&] {});
  return static_cast<SparseCsrTensorImpl*>(self.unsafeGetTensorImpl());
}

inline std::string layoutToString(
    Layout layout,
    bool upper = false,
    bool lower = false) {
  switch (layout) {
    case kSparseCsr:
      return (upper ? "CSR" : (lower ? "csr" : "Csr"));
    case kSparseCsc:
      return (upper ? "CSC" : (lower ? "csc" : "Csc"));
    case kSparseBsr:
      return (upper ? "BSR" : (lower ? "bsr" : "Bsr"));
    case kSparseBsc:
      return (upper ? "BSC" : (lower ? "bsc" : "Bsc"));
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

inline bool isCompressedRow(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout, "isCompressedRow", [&] { return true; }, [&] { return false; });
}

inline bool isCompressedColumn(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "isCompressedColumn",
      [&] { return false; },
      [&] { return true; });
}

inline std::string compressedIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "compressedIndicesName",
      [&] { return "crow_indices"; },
      [&] { return "ccol_indices"; });
}

inline std::string plainIndicesName(Layout layout) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      layout,
      "plainIndicesName",
      [&] { return "col_indices"; },
      [&] { return "row_indices"; });
}

inline std::string compressedDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "row";
    case kSparseCsc:
      return "column";
    case kSparseBsr:
      return "row block";
    case kSparseBsc:
      return "column block";
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return "";
  }
}

inline std::string plainDimName(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return "column";
    case kSparseCsc:
      return "row";
    case kSparseBsr:
      return "column block";
    case kSparseBsc:
      return "row block";
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

inline int compressedDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 2 : 1);
}

inline int plainDimension(
    Layout layout,
    IntArrayRef size,
    size_t dense_ndim = 0) {
  return size.size() - dense_ndim - (isCompressedRow(layout) ? 1 : 2);
}

inline int64_t numBatchDimensions(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "numBatchDimensions",
      [&self] { return self.crow_indices().dim() - 1; },
      [&self] { return self.ccol_indices().dim() - 1; });
}

inline std::pair<Tensor, Tensor> getCompressedPlainIndices(Tensor const& self) {
  return AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(
      self.layout(),
      "getCompressedPlainIndices",
      [&self] {
        return std::make_pair(self.crow_indices(), self.col_indices());
      },
      [&self] {
        return std::make_pair(self.ccol_indices(), self.row_indices());
      });
}

inline Layout flip_compressed_layout(Layout layout) {
  switch (layout) {
    case kSparseCsr:
      return kSparseCsc;
    case kSparseCsc:
      return kSparseCsr;
    case kSparseBsr:
      return kSparseBsc;
    case kSparseBsc:
      return kSparseBsr;
    default:
      TORCH_CHECK(false, "Not a sparse compressed layout:", layout);
      return kSparseCsr;
  }
}

} // namespace sparse_csr
} // namespace at
