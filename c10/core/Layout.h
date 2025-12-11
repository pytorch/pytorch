#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <torch/headeronly/core/Layout.h>

namespace c10 {

inline Layout layout_from_backend(Backend backend) {
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseMPS:
    case Backend::SparseHIP:
    case Backend::SparseVE:
    case Backend::SparseXPU:
    case Backend::SparsePrivateUse1:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    case Backend::SparseCsrCPU:
    case Backend::SparseCsrCUDA:
    case Backend::SparseCsrMPS:
    case Backend::SparseCsrHIP:
    case Backend::SparseCsrVE:
    case Backend::SparseCsrXPU:
      TORCH_CHECK(
          false,
          "Cannot map Backend SparseCsr(CPU|CUDA|HIP|VE|XPU|MPS) to a unique layout.");
    default:
      return Layout::Strided;
  }
  C10_DIAGNOSTIC_POP()
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
    case Layout::NumOptions:
    default:
      TORCH_CHECK(false, "Unknown layout");
  }
}

} // namespace c10
