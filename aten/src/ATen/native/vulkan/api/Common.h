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

#define VK_CHECK(function)              \
  do {                                  \
    const VkResult result = (function); \
    TORCH_CHECK(                        \
        VK_SUCCESS == result,           \
        C10_STRINGIZE(__FILE__),        \
        " [",                           \
        C10_STRINGIZE(__LINE__),        \
        "] "                            \
        "VkResult:",                    \
        result);                        \
  } while (false)

#define VK_CHECK_RELAXED(function)                          \
  do {                                                      \
    const VkResult result = (function);                     \
    TORCH_CHECK(VK_SUCCESS <= result, "VkResult:", result); \
  } while (false)

#endif /* USE_VULKAN_API */
