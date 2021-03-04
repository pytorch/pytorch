#pragma once

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

inline int64_t normalize(
    const int64_t dimension,
    const int64_t n) {
  return (dimension % n + n) % n;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
