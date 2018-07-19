#pragma once

#include <ATen/ScalarType.h>

namespace at {
enum class Layout { Strided, Sparse };

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
} // namespace at
