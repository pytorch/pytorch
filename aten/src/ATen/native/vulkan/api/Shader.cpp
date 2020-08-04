#include <ATen/native/vulkan/api/Shader.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <shaderc/shaderc.hpp>
#endif /* USE_VULKAN_SHADERC_RUNTIME */

namespace at {
namespace native {
namespace vulkan {
namespace api {

Shader::Descriptor::Descriptor(const Source& source)
 : type(Type::Source),
   // Intentionally zero initialize the uion.
   shader{.binary = {}} {
    shader.source = source;
}

Shader::Descriptor::Descriptor(const Binary& binary)
 : type(Type::Binary),
   shader{.binary = binary} {
}

#ifdef USE_VULKAN_SHADERC_RUNTIME

struct Shader::Factory::Compiler final {
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

  std::vector<const uint32_t> compile(const char* const source) const {
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

    return std::vector<const uint32_t>(result.cbegin(), result.cend());
  }
};

#else

struct Shader::Factory::Compiler final {
  std::vector<const uint32_t> compile(const char* const /* source */) const {
    return std::vector<const uint32_t>{};
  }
};

#endif /* USE_VULKAN_SHADERC_RUNTIME */

Shader::Factory::Factory(const VkDevice device)
 : device_(device),
   compiler_(new Compiler) {
}

// std::unique_ptr requires its template parameter to be fully defined.
// For that reason pimpl through unique_ptr requires the definition of
// the [default] constructor and assignment to show up after impl is
// fully defined.

Shader::Factory::Factory(Factory&&) = default;
Shader::Factory& Shader::Factory::Factory::operator=(Factory&&) = default;
Shader::Factory::~Factory() = default;

typename Shader::Factory::Handle Shader::Factory::operator()(
    const Descriptor& descriptor) const {
  std::vector<const uint32_t> binary;

  const uint32_t* code = nullptr;
  uint32_t count = 0u;

  if (Descriptor::Type::Source == descriptor.type) {
    binary = compiler_->compile(descriptor.shader.source.code);
    code = binary.data();
    count = binary.size();
  }
  else if (Descriptor::Type::Binary == descriptor.type) {
    code = descriptor.shader.binary.code;
    count = descriptor.shader.binary.count;
  }
  else {
    TORCH_INTERNAL_ASSERT(false, "Invalid descriptor type!");
  }

  const VkShaderModuleCreateInfo shader_module_create_info{
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    nullptr,
    0u,
    count * sizeof(uint32_t),
    code,
  };

  VkShaderModule shader_module{};
  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &shader_module));

  return Handle{
    shader_module,
    Deleter(device_),
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
