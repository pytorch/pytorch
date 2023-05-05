#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>

namespace at {
namespace native {
namespace vulkan {

struct GraphConfig final {
  api::ContextConfig context_config;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
