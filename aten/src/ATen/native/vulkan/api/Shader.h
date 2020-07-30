#pragma once

#include <ATen/native/vulkan/api/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {

struct Shader final {
  std::vector<uint32_t> binary;

  explicit Shader(const char* const glsl);
};

} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
