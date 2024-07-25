#pragma once

//
// Do NOT include vk_mem_alloc.h directly.
// Always include this file (Allocator.h) instead.
//

#include <ATen/native/vulkan/api/vk_api.h>
#include <c10/macros/Macros.h>

#ifdef USE_VULKAN_API

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

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wnullability-completeness")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-variable")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(
    "-Winconsistent-missing-destructor-override")
#include <include/vk_mem_alloc.h>
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()
C10_DIAGNOSTIC_POP()

#endif /* USE_VULKAN_API */
