#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace c10 {
enum class Layout : int8_t { Strided, SparseCOO, SparseGCS, Mkldnn, NumOptions };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparseCOO = Layout::SparseCOO;
constexpr auto kSparseGCS = Layout::SparseGCS;
constexpr auto kMkldnn = Layout::Mkldnn;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCOO_CPU:
    case Backend::SparseCOO_CUDA:
    case Backend::SparseHIP:
      return Layout::SparseCOO;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    default:
      return Layout::Strided;
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";
    case at::kSparseCOO:
      return stream << "SparseCOO";
    case at::kSparseGCS:
      return stream << "SparseGCS";
    case at::kMkldnn:
      return stream << "Mkldnn";
    default:
      AT_ERROR("Unknown layout");
  }
}

} // namespace c10
