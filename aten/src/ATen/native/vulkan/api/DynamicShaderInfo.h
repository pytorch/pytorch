#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/native/vulkan/api/Shader.h>
#include <c10/macros/Macros.h>

namespace at::native::vulkan::api {

// Owning wrapper for ShaderInfo.
class TORCH_API DynamicShaderInfo final {
 public:
  DynamicShaderInfo() = default;
  explicit DynamicShaderInfo(
      std::string name,
      std::unique_ptr<const std::uint32_t[]> spirv_bin,
      const std::uint32_t spirv_size,
      std::vector<VkDescriptorType> layout)
      : repr_(std::move(name), spirv_bin.get(), spirv_size, std::move(layout)),
        src_code_bin_owner_(std::move(spirv_bin)) {}

  // REVIEW: I don't think dynamic shaders will require tile size or
  // bias/weight storage types.

  const ShaderInfo& shader_info() const {
    return repr_;
  }

 private:
  ShaderInfo repr_;
  std::unique_ptr<const uint32_t[]> src_code_bin_owner_;
};

TORCH_API DynamicShaderInfo
compile_glsl(std::string name, std::string_view src);
} // namespace at::native::vulkan::api
