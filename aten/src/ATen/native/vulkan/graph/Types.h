#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

/*
 * This class is modelled after c10::IValue; however, it is simplified and does
 * not support as many types. However, the core design is the same; it is a
 * tagged union over the types supported by the Vulkan Graph type.
 */
enum class TypeTag : uint32_t { NONE, TENSOR, STAGING, INT, DOUBLE, BOOL };

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
