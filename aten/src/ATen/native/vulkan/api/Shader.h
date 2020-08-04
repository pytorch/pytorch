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
    } type;

    union {
      struct {
        const char* glsl; // Null-terminated
        uint32_t unused;  // Padding
      } source;

      struct {
        const uint32_t* spirv;
        uint32_t size;    // Bytes
      } binary;
    } shader;

    Descriptor(const char* glsl);
    Descriptor(const uint32_t* spirv, uint32_t bytes);

    inline bool operator==(const Descriptor& descriptor) const {
      static_assert(
          sizeof(descriptor.shader.source) == sizeof(descriptor.shader.binary),
          "This implementation requires sizeof(Source) to be equal to sizeof(Binary).");

      return (type == descriptor.type) &&
             (shader.binary.spirv == descriptor.shader.binary.spirv) &&
             (shader.binary.size == descriptor.shader.binary.size);
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
        static_assert(
            sizeof(descriptor.shader.source) == sizeof(descriptor.shader.binary),
            "This implementation requires sizeof(Source) to be equal to sizeof(Binary).");

        return c10::get_hash(
            descriptor.type,
            descriptor.shader.binary.spirv,
            descriptor.shader.binary.size);
      }
    };

    Handle operator()(const Descriptor& descriptor) const;

   private:
    VkDevice device_;
    struct Compiler;
    std::unique_ptr<Compiler> compiler_;
  };

  struct WorkGroup final {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    inline bool operator==(const WorkGroup& work_group) const {
      return (x == work_group.x) && (y == work_group.y) && (z == work_group.z);
    }
  };

  typedef api::Cache<Factory> Cache;
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
