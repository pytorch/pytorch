#pragma once

#include <ATen/native/vulkan/api/Common.h>

#ifdef DEBUG
  #define VMA_DEBUG_LOG(format, ...)  \
    do {                              \
      printf(format, ##__VA_ARGS__);  \
      printf("\n");                   \
    } while(false)
#endif /* DEBUG */

#ifdef __clang__
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wnullability-completeness"
  #pragma clang diagnostic ignored "-Wunused-variable"
#endif /* __clang__ */

// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.

#include <ATen/native/vulkan/api/vk_mem_alloc.h>

#ifdef __clang__
  #pragma clang diagnostic pop
#endif /* __clang__ */
