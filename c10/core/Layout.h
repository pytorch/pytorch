#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <ostream>

namespace c10 {
enum class Layout : int8_t {
  Strided,
  Sparse,
  SparseCsr,
  Mkldnn,
  SparseCsc,
  SparseBsr,
  SparseBsc,
  Jagged,
  NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparseXPU:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
      TORCH_CHECK(
          false,
          "Cannot map Backend SparseCsrCPU|SparseCsrCUDA to a unique layout.");
    default:
      return Layout::Strided;
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";
    case at::kSparse:
      return stream << "Sparse";
    case at::kSparseCsr:
      return stream << "SparseCsr";
    case at::kSparseCsc:
      return stream << "SparseCsc";
    case at::kSparseBsr:
      return stream << "SparseBsr";
    case at::kSparseBsc:
      return stream << "SparseBsc";
    case at::kMkldnn:
      return stream << "Mkldnn";
    case at::kJagged:
      return stream << "Jagged";
    default:
      TORCH_CHECK(false, "Unknown layout");
  }
}

} // namespace c10
