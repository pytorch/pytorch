#include <ATen/native/vulkan/api/Shader.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <shaderc/shaderc.hpp>
#endif /* USE_VULKAN_SHADERC_RUNTIME */

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace api {

struct Shader::Cache::Compiler final {
#ifdef USE_VULKAN_SHADERC_RUNTIME
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

  Binary compile(const char* const source) {
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

    return Binary{
      {result.cbegin(), result.cend()},
    };
  }
#else
  Binary compile(const char* const /* source */) {
    return Binary{};
  }
#endif /* USE_VULKAN_SHADERC_RUNTIME */
};

Shader::Cache::Cache()
  : compiler_(new Compiler) {
}

Shader Shader::Cache::retrieve(
    const char* const key,
    const char* const glsl) {
  auto iterator = shaders_.find(key);
  if (C10_UNLIKELY(shaders_.cend() != iterator)) {
    iterator = shaders_.insert({key, compiler_->compile(glsl)}).first;
  }

  return Shader{
      iterator->second.data.data(),
      iterator->second.data.size(),
    };;
}

} // namespace api
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
