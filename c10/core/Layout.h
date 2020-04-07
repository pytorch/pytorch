#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace c10 {
enum class Layout : int8_t { Strided, Sparse, Mkldnn, Vulkan };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kVulkan = Layout::Vulkan;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return Layout::Sparse;
    case Backend::MkldnnCPU:
      return Layout::Mkldnn;
    case Backend::Vulkan:
      return Layout::Vulkan;
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
    case at::kMkldnn:
      return stream << "Mkldnn";
    case at::kVulkan:
      return stream << "Vulkan";
    default:
      AT_ERROR("Unknown layout");
  }
}

} // namespace c10
