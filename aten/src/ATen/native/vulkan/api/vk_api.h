#pragma once

#ifdef USE_VULKAN_API

#ifdef USE_VULKAN_WRAPPER
#ifdef USE_VULKAN_VOLK
#include <volk.h>
#else
#include <vulkan_wrapper.h>
#endif /* USE_VULKAN_VOLK */
#else
#include <vulkan/vulkan.h>
#endif /* USE_VULKAN_WRAPPER */

#endif /* USE_VULKAN_API */
