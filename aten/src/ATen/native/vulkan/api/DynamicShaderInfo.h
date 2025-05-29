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
      int num_layouts_in_src)
      : repr_(std::move(name), spirv_bin.get(), spirv_size, {}),
        num_layouts_in_src_(num_layouts_in_src),
        src_code_bin_owner_(std::move(spirv_bin)) {}

  const ShaderInfo& shader_info() const {
    return repr_;
  }

  int get_expected_number_of_arguments() const {
    return num_layouts_in_src_;
  }

  bool layout_is_initialized() const {
    return !repr_.kernel_layout.empty();
  }

  void set_layout(ShaderLayout::Signature layout) {
    repr_.kernel_layout = std::move(layout);
  }

 private:
  ShaderInfo repr_;
  int num_layouts_in_src_;
  std::unique_ptr<const uint32_t[]> src_code_bin_owner_;
};

TORCH_API DynamicShaderInfo
compile_glsl(std::string name, std::string_view src, bool use_buffers);
} // namespace at::native::vulkan::api
