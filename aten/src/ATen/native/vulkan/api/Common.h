#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/native/vulkan/spv.h>

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
#define VK_KERNEL(name)                          \
  ::at::native::vulkan::api::ShaderSource{       \
    #name,                                       \
    name##_spv,                                  \
    name##_spv_len,                              \
  }
#else
#define VK_KERNEL(name)                          \
  ::at::native::vulkan::api::ShaderSource{       \
    name##_spv,                                  \
    name##_spv_len,                              \
  }
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

#ifdef USE_VULKAN_WRAPPER
#ifdef USE_VULKAN_VOLK
#include <volk.h>
#else
#include <vulkan_wrapper.h>
#endif /* USE_VULKAN_VOLK */
#else
#include <vulkan/vulkan.h>
#endif /* USE_VULKAN_WRAPPER */

#define VK_CHECK(function)                                  \
  do {                                                      \
    const VkResult result = (function);                     \
    TORCH_CHECK(                                            \
        VK_SUCCESS == result,                               \
        C10_STRINGIZE(__FILE__), " [",                      \
        C10_STRINGIZE(__LINE__), "] "                       \
        "VkResult:", result);                               \
  } while (false)

#define VK_CHECK_RELAXED(function)                          \
  do {                                                      \
    const VkResult result = (function);                     \
    TORCH_CHECK(VK_SUCCESS <= result, "VkResult:", result); \
  } while (false)

#endif /* USE_VULKAN_API */
