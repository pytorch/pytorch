#pragma once

#ifdef USE_VULKAN_API

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace utils {

inline uint32_t div_up(
    const uint32_t numerator,
    const uint32_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

inline VkFormat convert(const caffe2::TypeMeta dtype) {
  switch (c10::typeMetaToScalarType(dtype)) {
    case kFloat:
#ifdef VULKAN_FP16_INFERENCE
      return VK_FORMAT_R16G16B16A16_SFLOAT;
#else
      return VK_FORMAT_R32G32B32A32_SFLOAT;
#endif /* VULKAN_FP16_INFERENCE */

    default:
      TORCH_CHECK(
        false,
        "Vulkan tensor format not supported!");
  }

  return VK_FORMAT_UNDEFINED;
}

} // namespace utils
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
