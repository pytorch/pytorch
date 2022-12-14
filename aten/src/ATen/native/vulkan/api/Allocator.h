#pragma once

//
// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.
//

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>

#define VMA_VULKAN_VERSION 1000000

#ifdef USE_VULKAN_WRAPPER
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#else
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif /* USE_VULKAN_WRAPPER */

#define VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE (32ull * 1024 * 1024)
#define VMA_SMALL_HEAP_MAX_SIZE (256ull * 1024 * 1024)

#define VMA_STATS_STRING_ENABLED 0

#ifdef VULKAN_DEBUG
#define VMA_DEBUG_ALIGNMENT 4096
#define VMA_DEBUG_ALWAYS_DEDICATED_MEMORY 0
#define VMA_DEBUG_DETECT_CORRUPTION 1
#define VMA_DEBUG_GLOBAL_MUTEX 1
#define VMA_DEBUG_INITIALIZE_ALLOCATIONS 1
#define VMA_DEBUG_MARGIN 64
#define VMA_DEBUG_MIN_BUFFER_IMAGE_GRANULARITY 256
#define VMA_RECORDING_ENABLED 1

#define VMA_DEBUG_LOG(format, ...)
/*
#define VMA_DEBUG_LOG(format, ...) do { \
    printf(format, __VA_ARGS__); \
    printf("\n"); \
} while(false)
*/
#endif /* VULKAN_DEBUG */

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif /* __clang__ */

#include <include/vk_mem_alloc.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif /* __clang__ */

#endif /* USE_VULKAN_API */
