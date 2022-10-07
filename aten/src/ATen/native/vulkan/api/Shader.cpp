#include <ATen/native/vulkan/api/Shader.h>

#ifdef USE_VULKAN_SHADERC_RUNTIME
#include <shaderc/shaderc.hpp>
#endif /* USE_VULKAN_SHADERC_RUNTIME */

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// ShaderSource
//

ShaderSource::ShaderSource()
    : type(ShaderSource::Type::SPIRV),
      src_code{
          .spirv =
              {
                  nullptr,
                  0u,
              },
      } {}

ShaderSource::ShaderSource(std::string name, const char* const glsl_src)
    : type(ShaderSource::Type::GLSL),
      src_code{
          .glsl =
              {
                  glsl_src,
                  0u,
              },
      },
      kernel_name{std::move(name)} {}

ShaderSource::ShaderSource(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    const std::vector<VkDescriptorType>& layout)
    : type(Type::SPIRV),
      src_code{
          .spirv =
              {
                  spirv_bin,
                  size,
              },
      },
      kernel_name{std::move(name)},
      kernel_layout{layout} {}

bool operator==(const ShaderSource& _1, const ShaderSource& _2) {
  if (_1.type != _2.type) {
    return false;
  }

  if (_1.type == ShaderSource::Type::SPIRV) {
    return (
        _1.src_code.spirv.bin == _2.src_code.spirv.bin &&
        _1.src_code.spirv.size == _2.src_code.spirv.size);
  } else {
    return (_1.src_code.glsl.src == _2.src_code.glsl.src);
  }
}

//
// ShaderLayout
//

ShaderLayout::ShaderLayout(
    const VkDevice device,
    const ShaderLayout::Signature& signature)
    : device_(device), handle_{VK_NULL_HANDLE} {
  c10::SmallVector<VkDescriptorSetLayoutBinding, 6u> bindings;

  uint32_t binding_num = 0u;
  for (const VkDescriptorType type : signature) {
    bindings.push_back({
        binding_num++, // binding
        type, // descriptorType
        1u, // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
        nullptr, // pImmutableSamplers
    });
  }

  const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(bindings.size()), // bindingCount
      bindings.data(), // pBindings
  };

  VK_CHECK(vkCreateDescriptorSetLayout(
      device_, &descriptor_set_layout_create_info, nullptr, &handle_));
}

ShaderLayout::ShaderLayout(ShaderLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderLayout::~ShaderLayout() {
  if C10_LIKELY (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDescriptorSetLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkDescriptorSetLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderModule
//

ShaderModule::ShaderModule(const VkDevice device, const ShaderSource& source)
    : device_(device), handle_{VK_NULL_HANDLE} {
  const uint32_t* code = source.src_code.spirv.bin;
  uint32_t size = source.src_code.spirv.size;

  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // codeSize
      code, // pCode
  };

  VK_CHECK(vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_));
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderModule::~ShaderModule() {
  if C10_LIKELY (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyShaderModule(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderLayoutCache
//

ShaderLayoutCache::ShaderLayoutCache(const VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
  cache_ = std::move(other.cache_);
}

ShaderLayoutCache::~ShaderLayoutCache() {
  purge();
}

VkDescriptorSetLayout ShaderLayoutCache::retrieve(
    const ShaderLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if C10_UNLIKELY (cache_.cend() == it) {
    it = cache_.insert({key, ShaderLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ShaderCache
//

ShaderCache::ShaderCache(const VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
  cache_ = std::move(other.cache_);
}

ShaderCache::~ShaderCache() {
  purge();
}

VkShaderModule ShaderCache::retrieve(const ShaderCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if C10_UNLIKELY (cache_.cend() == it) {
    it = cache_.insert({key, ShaderCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderCache::purge() {
  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
