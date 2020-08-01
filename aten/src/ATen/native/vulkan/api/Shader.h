#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Shader final {
  struct Descriptor final {
    const uint32_t* code;
    uint32_t count;
  };

  class Factory final {
   public:
    explicit Factory(VkDevice device);

    typedef Shader::Descriptor Descriptor;
    typedef VK_DELETER(ShaderModule) Deleter;
    typedef Handle<VkShaderModule, Deleter> Handle;

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
  };

  class Cache final {
   public:
    explicit Cache(VkDevice device);
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = default;
    Cache& operator=(Cache&&) = default;
    ~Cache() = default;

    VkShaderModule retrieve(const char* key, const char* source = nullptr);
    VkShaderModule retrieve(const char* key, const Descriptor* descriptor = nullptr);

   private:
    struct Compiler;
    std::unique_ptr<Compiler> compiler_;
    api::Cache<const char*, VkShaderModule, Factory> cache_;
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
