#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/impl/Registry.h>
#include <ATen/native/vulkan/spv.h>

namespace at {
namespace native {
namespace vulkan {

const api::ShaderInfo& get_shader_info(const std::string& shader_name) {
  const ShaderListing::const_iterator shader_infos_iterator =
      get_shader_infos().find(shader_name);

  VK_CHECK_COND(
      shader_infos_iterator != get_shader_infos().end(),
      "Could not get ShaderInfo named ",
      shader_name);

  return shader_infos_iterator->second;
}

const api::ShaderInfo& look_up_shader_info(const std::string& op_name) {
  const ShaderRegistry::iterator registry_iterator =
      get_shader_registry().find(op_name);

  VK_CHECK_COND(
      registry_iterator != get_shader_registry().end(),
      "Could not look up ShaderInfo for ",
      op_name,
      " in shader registry");

  const RegistryKeyMap& registry_key_map = registry_iterator->second;

  // Look for "override" and "catchall" keys
  for (const std::string key : {"override", "catchall"}) {
    const RegistryKeyMap::const_iterator registry_key_iterator =
        registry_key_map.find(key);
    if (registry_key_iterator != registry_key_map.end()) {
      const ShaderListing::const_iterator shader_infos_iterator =
          get_shader_infos().find(registry_key_iterator->second);

      VK_CHECK_COND(
          shader_infos_iterator != get_shader_infos().end(),
          "Could not get ShaderInfo named ",
          registry_key_iterator->second,
          " (listed under ",
          op_name,
          " -> ",
          key,
          " in shader registry)");

      return shader_infos_iterator->second;
    }
  }

  VK_CHECK_COND(
      false,
      "Could not look up ShaderInfo for ",
      op_name,
      " with a valid key in shader registry");
}

void set_registry_override(
    const std::string& op_name,
    const std::string& shader_name) {
  const ShaderRegistry::iterator registry_iterator =
      get_shader_registry().find(op_name);

  VK_CHECK_COND(
      registry_iterator != get_shader_registry().end(),
      "Could not look up ShaderInfo for ",
      op_name,
      " in shader registry");

  VK_CHECK_COND(
      get_shader_infos().find(shader_name) != get_shader_infos().end(),
      "Could not get ShaderInfo named ",
      shader_name);

  registry_iterator->second["override"] = shader_name;
}

} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
