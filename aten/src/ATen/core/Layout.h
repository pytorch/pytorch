#pragma once

#include <ATen/core/Backend.h>
#include <ATen/core/Error.h>

#include <iostream>

namespace at {
enum class Layout : int8_t { Strided, Sparse };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
      return Layout::Sparse;
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
    default:
      AT_ERROR("Unknown layout");
  }
}

} // namespace at
