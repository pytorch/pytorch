#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>

#include <ATen/native/vulkan/api/vk_api.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#define VK_KERNEL(name)                     \
  ::at::native::vulkan::api::ShaderSource { \
#name, name##_glsl,                     \
  }
#else
#include <ATen/native/vulkan/spv.h>
#define VK_KERNEL(name)                                  \
  ::at::native::vulkan::api::ShaderSource {              \
#name, name##_spv, name##_spv_len, name##_spv_layout \
  }
#endif /* USE_VULKAN_SHADERC_RUNTIME */

/*
 * Check that the return code of a Vulkan API call is VK_SUCCESS, throwing an
 * error with the returned code if not. Note that TORCH_CHECK is not used here
 * because error messages are stripped in production builds, and the VkResult
 * should be preserved in the case of failure to make debugging easier.
 */
#define VK_CHECK(function)                                            \
  do {                                                                \
    const VkResult result = (function);                               \
    if (VK_SUCCESS != result) {                                       \
      throw c10::Error(                                               \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},      \
          c10::str("Expected VK_SUCCESS, got VkResult of ", result)); \
    }                                                                 \
  } while (false)

/*
 * Throw an error tracking the file, line, and function that called VK_THROW.
 */
#define VK_THROW(msg)                                                          \
  do {                                                                         \
    throw c10::Error(                                                          \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, c10::str(msg)); \
  } while (false)

#endif /* USE_VULKAN_API */
