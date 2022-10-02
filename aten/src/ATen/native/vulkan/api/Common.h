#pragma once

#ifdef USE_VULKAN_API

#include <c10/util/Exception.h>
#include <utility>

#include <ATen/native/vulkan/api/vk_api.h>

#define CONCAT_LITERALS(a, b) #a #b
#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <ATen/native/vulkan/glsl.h>
#define VK_KERNEL(name)                          \
  ::at::native::vulkan::api::ShaderSource {      \
    CONCAT_LITERALS(vulkan., name), name##_glsl, \
  }
#else
#include <ATen/native/vulkan/spv.h>
#define VK_KERNEL(name)                                         \
  ::at::native::vulkan::api::ShaderSource {                     \
    CONCAT_LITERALS(vulkan., name), name##_spv, name##_spv_len, \
        name##_spv_layout                                       \
  }
#endif /* USE_VULKAN_SHADERC_RUNTIME */

/*
 * Check that the return code of a Vulkan API call is VK_SUCCESS, throwing an
 * error with the returned code if not. If STRIP_ERROR_MESSAGES is defined then
 * only the return code will be preserved.
 */
#ifdef STRIP_ERROR_MESSAGES
#define VK_CHECK(function)                                       \
  do {                                                           \
    const VkResult result = (function);                          \
    if (VK_SUCCESS != result) {                                  \
      throw c10::Error(                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          c10::str(result));                                     \
    }                                                            \
  } while (false)
#else
#define VK_CHECK(function)                                       \
  do {                                                           \
    const VkResult result = (function);                          \
    if (VK_SUCCESS != result) {                                  \
      throw c10::Error(                                          \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          c10::str(                                              \
              C10_STRINGIZE(__FILE__),                           \
              "[",                                               \
              C10_STRINGIZE(__LINE__),                           \
              "] Expected VK_SUCCESS, got VkResult of ",         \
              result));                                          \
    }                                                            \
  } while (false)
#endif /* STRIP_ERROR_MESSAGES */

#endif /* USE_VULKAN_API */
