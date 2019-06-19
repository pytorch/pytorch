#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace c10 {
enum class Layout : int8_t { Strided, Sparse, Mkldnn };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kMkldnn = Layout::Mkldnn;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    default:
      return Layout::Strided;
  }
}

inline const char* toString(at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return "Strided";
    case at::kSparse:
      return "Sparse";
    case at::kMkldnn:
      return "Mkldnn";
    default:
      AT_ERROR("Unknown layout");
  }
}

} // namespace c10
