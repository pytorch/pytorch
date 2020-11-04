#pragma once

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {

inline uint32_t div_up(
    const uint32_t numerator,
    const uint32_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
