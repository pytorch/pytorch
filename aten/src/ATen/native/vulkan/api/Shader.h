#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Shader final {
  struct Descriptor final {
    enum class Type {
      Source,
      Binary,
    };

    struct Source final {
      const char* code; /* null terminated */
    };

    struct Binary final {
      const uint32_t* code;
      uint32_t count; /* in uints of uint32_t, not bytes */
    };

    Type type;

    union {
      Source source;
      Binary binary;
    } shader;

    Descriptor() = delete;
    explicit Descriptor(const Source& source);
    explicit Descriptor(const Binary& binary);

    inline bool operator==(const Descriptor& descriptor) const {
      return (type == descriptor.type) &&
             // We have zero initialized the unused portion of the union in the
             // constructor to make this comparison work for descriptor.type == Source.
             (shader.binary.code == descriptor.shader.binary.code) &&
             (shader.binary.count == descriptor.shader.binary.count);
    }
  };

  class Factory final {
   public:
    explicit Factory(VkDevice device);
    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;
    Factory(Factory&&);
    Factory& operator=(Factory&&);
    ~Factory();

    typedef Shader::Descriptor Descriptor;
    typedef VK_DELETER(ShaderModule) Deleter;
    typedef Handle<VkShaderModule, Deleter> Handle;

    struct Hasher {
      inline size_t operator()(const Descriptor& descriptor) const {
          return c10::get_hash(
              descriptor.type,
              descriptor.shader.binary.code,
              descriptor.shader.binary.count);
      }
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    struct Compiler;
    std::unique_ptr<Compiler> compiler_;
  };

  typedef api::Cache<Factory> Cache;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
