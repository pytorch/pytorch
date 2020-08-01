#include <ATen/native/vulkan/api/Shader.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <shaderc/shaderc.hpp>
#endif /* USE_VULKAN_SHADERC_RUNTIME */

namespace at {
namespace native {
namespace vulkan {
namespace api {

Shader::Factory::Factory(const VkDevice device)
 : device_(device) {
}

typename Shader::Factory::Handle Shader::Factory::operator()(const Descriptor& descriptor) const {
  VkShaderModuleCreateInfo create_info{};

  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.pCode = descriptor.code;
  create_info.codeSize = descriptor.count * sizeof(uint32_t);

  VkShaderModule shader_module{};

  VK_CHECK(vkCreateShaderModule(
      device_, &create_info, nullptr, &shader_module));

  return Handle{
    shader_module,
    Deleter(device_),
  };
}

#ifdef USE_VULKAN_SHADERC_RUNTIME

struct Shader::Cache::Compiler final {
  shaderc::Compiler context;
  shaderc::CompileOptions options;

  Compiler() {
    options.SetSourceLanguage(shaderc_source_language_glsl);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
    options.SetWarningsAsErrors();
  #ifdef DEBUG
    options.SetGenerateDebugInfo();
    options.SetOptimizationLevel(shaderc_optimization_level_zero);
  #else
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
  #endif /* DEBUG */
  }

  std::vector<uint32_t> compile(const char* const source) {
    const shaderc::SpvCompilationResult result = context.CompileGlslToSpv(
        source,
        ::strlen(source),
        shaderc_compute_shader,
        "vulkan_shader.comp",
        options);

    const shaderc_compilation_status status = result.GetCompilationStatus();
    TORCH_INTERNAL_ASSERT(
        shaderc_compilation_status_success == status,
        "Shader compilation error: ",
        result.GetErrorMessage());

    return std::vector<uint32_t>(result.cbegin(), result.cend());
  }
};

#else

struct Shader::Cache::Compiler final {
  std::vector<uint32_t> compile(const char* const /* source */) {
    return std::vector<uint32_t>{};
  }
};

#endif /* USE_VULKAN_SHADERC_RUNTIME */

Shader::Cache::Cache(const VkDevice device)
 : compiler_(new Compiler),
   cache_(Factory(device)) {
}

VkShaderModule Shader::Cache::retrieve(
    const char* const key,
    const char* const source) {
  const VkShaderModule shader_module = cache_.retrieve(key);
  if (VK_NULL_HANDLE != shader_module) {
    return shader_module;
  }

  const std::vector<uint32_t> binary = compiler_->compile(source);
  const Descriptor descriptor{
      binary.data(),
      binary.size()
  };

  return retrieve(key, &descriptor);
}

VkShaderModule Shader::Cache::retrieve(
    const char* const key,
    const Descriptor* const descriptor) {
  return cache_.retrieve(key, descriptor);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
