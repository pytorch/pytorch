#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>

#include <mutex>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class ShaderLayout final {
 public:
  using Signature = c10::SmallVector<VkDescriptorType, 6u>;

  explicit ShaderLayout(const VkDevice, const Signature&);

  ShaderLayout(const ShaderLayout&) = delete;
  ShaderLayout& operator=(const ShaderLayout&) = delete;

  ShaderLayout(ShaderLayout&&) noexcept;
  ShaderLayout& operator=(ShaderLayout&&) = delete;

  ~ShaderLayout();

 private:
  VkDevice device_;
  VkDescriptorSetLayout handle_;

 public:
  VkDescriptorSetLayout handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept;
};

struct ShaderSource final {
  enum class Type { GLSL, SPIRV } type;

  union {
    struct {
      const char* src; // Null-terminated
      uint32_t unused; // padding
    } glsl;
    struct {
      const uint32_t* bin;
      uint32_t size;
    } spirv;
  } src_code;

  std::string kernel_name{""};
  ShaderLayout::Signature kernel_layout{};

  // Shader Metadata
  utils::uvec3 out_tile_size{1u, 1u, 1u};

  explicit ShaderSource();
  explicit ShaderSource(std::string, const char*);
  explicit ShaderSource(
      std::string,
      const uint32_t*,
      const uint32_t,
      const std::vector<VkDescriptorType>&);
};

bool operator==(const ShaderSource& _1, const ShaderSource& _2);

struct ShaderInfo final {
  ShaderSource shader_src;
  c10::SmallVector<uint32_t, 4> tile_size;
  StorageType weight_storage_type{StorageType::UNKNOWN};

  explicit ShaderInfo() = default;
  explicit ShaderInfo(std::string, const char*);
  explicit ShaderInfo(
      std::string,
      const uint32_t*,
      const uint32_t,
      const std::vector<VkDescriptorType>&,
      const std::vector<uint32_t>& tile_size,
      const StorageType weight_storage_type);
};

class ShaderModule final {
 public:
  explicit ShaderModule(const VkDevice device, const ShaderSource& source);

  ShaderModule(const ShaderModule&) = delete;
  ShaderModule& operator=(const ShaderModule&) = delete;

  ShaderModule(ShaderModule&&) noexcept;
  ShaderModule& operator=(ShaderModule&&) = delete;

  ~ShaderModule();

 private:
  VkDevice device_;
  VkShaderModule handle_;

 public:
  inline VkShaderModule handle() const {
    return handle_;
  }

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept;
};

class ShaderLayoutCache final {
 public:
  explicit ShaderLayoutCache(const VkDevice device);

  ShaderLayoutCache(const ShaderLayoutCache&) = delete;
  ShaderLayoutCache& operator=(const ShaderLayoutCache&) = delete;

  ShaderLayoutCache(ShaderLayoutCache&&) noexcept;
  ShaderLayoutCache& operator=(ShaderLayoutCache&&) = delete;

  ~ShaderLayoutCache();

  using Key = ShaderLayout::Signature;
  using Value = ShaderLayout;

  struct Hasher {
    inline size_t operator()(const ShaderLayout::Signature& signature) const {
      size_t hashed = 0u;

      for (const VkDescriptorType type : signature) {
        hashed = c10::hash_combine(hashed, c10::get_hash(type));
      }

      return hashed;
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  ska::flat_hash_map<Key, Value, Hasher> cache_;

 public:
  VkDescriptorSetLayout retrieve(const Key&);
  void purge();
};

class ShaderCache final {
 public:
  explicit ShaderCache(const VkDevice device);

  ShaderCache(const ShaderCache&) = delete;
  ShaderCache& operator=(const ShaderCache&) = delete;

  ShaderCache(ShaderCache&&) noexcept;
  ShaderCache& operator=(ShaderCache&&) = delete;

  ~ShaderCache();

  using Key = ShaderSource;
  using Value = ShaderModule;

  struct Hasher {
    inline size_t operator()(const ShaderSource& source) const {
      return c10::get_hash(
          source.type, source.src_code.spirv.bin, source.src_code.spirv.size);
    }
  };

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  ska::flat_hash_map<Key, Value, Hasher> cache_;

 public:
  VkShaderModule retrieve(const Key&);
  void purge();
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

inline bool operator==(
    const VkDescriptorSetLayoutBinding& _1,
    const VkDescriptorSetLayoutBinding& _2) {
  return (
      _1.binding == _2.binding && _1.descriptorType == _2.descriptorType &&
      _1.descriptorCount == _2.descriptorCount &&
      _1.stageFlags == _2.stageFlags &&
      _1.pImmutableSamplers == _2.pImmutableSamplers);
}

#endif /* USE_VULKAN_API */
