#pragma once

#include <ATen/native/vulkan/api/Common.h>

#ifdef __clang__
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wnullability-completeness"
  #pragma clang diagnostic ignored "-Wunused-variable"
#endif

// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.

#include <ATen/native/vulkan/api/vk_mem_alloc.h>

#ifdef __clang__
  #pragma clang diagnostic pop
#endif
