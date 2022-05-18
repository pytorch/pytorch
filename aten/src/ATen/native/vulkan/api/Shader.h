#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct ShaderSource final {
  enum class Type {
    GLSL,
    SPIRV
  } type;

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

  ShaderSource(const char* glsl);
  ShaderSource(const uint32_t* spirv, uint32_t bytes);
};

class ShaderLayout final {
 public:
  typedef c10::SmallVector<VkDescriptorType, 6u> Signature;

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

  struct Hasher {
    size_t operator()(const Signature&) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderLayout& lhs, ShaderLayout& rhs);
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

  struct Hasher {
    size_t operator()(const ShaderSource&) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ShaderModule& lhs, ShaderModule& rhs);
};

class ShaderLayoutCache final {
 public:
  explicit ShaderLayoutCache(const VkDevice device);

  ShaderLayoutCache(const ShaderLayoutCache&) = delete;
  ShaderLayoutCache& operator=(const ShaderLayoutCache&) = delete;

  ShaderLayoutCache(ShaderLayoutCache&&) = default;
  ShaderLayoutCache& operator=(ShaderLayoutCache&&) = delete;

  ~ShaderLayoutCache();

  typedef ShaderLayout::Signature Key;
  typedef ShaderLayout Value;
  typedef ShaderLayout::Hasher Hasher;

 private:
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

  ShaderCache(ShaderCache&&) = default;
  ShaderCache& operator=(ShaderCache&&) = delete;

  ~ShaderCache();

  typedef ShaderSource Key;
  typedef ShaderModule Value;
  typedef ShaderModule::Hasher Hasher;

 private:
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

  return (_1.binding == _2.binding && \
          _1.descriptorType == _2.descriptorType && \
          _1.descriptorCount == _2.descriptorCount && \
          _1.stageFlags == _2.stageFlags && \
          _1.pImmutableSamplers == _2.pImmutableSamplers);
}

#endif /* USE_VULKAN_API */
