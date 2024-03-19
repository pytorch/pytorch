#include <ATen/native/vulkan/api/ShaderRegistry.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

bool ShaderRegistry::has_shader(const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);
  return it != listings_.end();
}

bool ShaderRegistry::has_dispatch(const std::string& op_name) {
  const Registry::const_iterator it = registry_.find(op_name);
  return it != registry_.end();
}

void ShaderRegistry::register_shader(ShaderInfo&& shader_info) {
  if (has_shader(shader_info.kernel_name)) {
    VK_THROW(
        "Shader with name ", shader_info.kernel_name, "already registered");
  }
  listings_.emplace(shader_info.kernel_name, shader_info);
}

void ShaderRegistry::register_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  if (!has_dispatch(op_name)) {
    registry_.emplace(op_name, Dispatcher());
  }
  const Dispatcher::const_iterator it = registry_[op_name].find(key);
  if (it != registry_[op_name].end()) {
    registry_[op_name][key] = shader_name;
  } else {
    registry_[op_name].emplace(key, shader_name);
  }
}

const ShaderInfo& ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);

  VK_CHECK_COND(
      it != listings_.end(),
      "Could not find ShaderInfo with name ",
      shader_name);

  return it->second;
}

ShaderRegistry& shader_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
