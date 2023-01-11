#pragma once

#ifdef USE_VULKAN_API

#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace api {
// Forward declaration of ShaderInfo
struct ShaderInfo;
} // namespace api

/**
 * Get the shader with a given name
 */
const api::ShaderInfo& get_shader_info(const std::string& shader_name);

/**
 * Look up which shader to use for a given op in the shader registry
 */
const api::ShaderInfo& look_up_shader_info(const std::string& op_name);

void set_registry_override(
    const std::string& op_name,
    const std::string& shader_name);

} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
