#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Shader final {
  /*
    Descriptor
  */

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

  /*
    Factory
  */

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

  /*
    Cache
  */

  typedef api::Cache<Factory> Cache;

  //
  // Layout
  //

  struct Layout final {
    /*
      Descriptor
    */

    struct Descriptor final {
      struct Slot final {
        VkDescriptorSetLayoutBinding binding;

        inline bool operator==(const Slot& slot) const {
          return (binding.binding == slot.binding.binding) &&
                 (binding.descriptorType == slot.binding.descriptorType) &&
                 (binding.descriptorCount == slot.binding.descriptorCount) &&
                 (binding.stageFlags == slot.binding.stageFlags) &&
                 (binding.pImmutableSamplers == slot.binding.pImmutableSamplers);
        }

        inline size_t hash() const {
          return c10::get_hash(
              binding.binding,
              binding.descriptorType,
              binding.descriptorCount,
              binding.stageFlags,
              binding.pImmutableSamplers);
        }
      };

      std::array<Slot, 8u> slots;

      inline bool operator==(const Descriptor& descriptor) const {
        return (slots == descriptor.slots);
      }
    };

    /*
      Factory
    */

    class Factory final {
     public:
      explicit Factory(VkDevice device);

      typedef Layout::Descriptor Descriptor;
      typedef VK_DELETER(DescriptorSetLayout) Deleter;
      typedef Handle<VkDescriptorSetLayout, Deleter> Handle;

      struct Hasher {
        inline size_t operator()(const Descriptor& descriptor) const {
          size_t hash = 0u;

          for (const Descriptor::Slot& slot : descriptor.slots) {
            hash = c10::hash_combine(hash, slot.hash());
          }

          return hash;
        }
      };

      Handle operator()(const Descriptor& descriptor) const;

     private:
      VkDevice device_;
    };

    /*
      Cache
    */

    typedef api::Cache<Factory> Cache;
  };

  //
  // Work Group
  //

  struct WorkGroup final {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    inline bool operator==(const WorkGroup& work_group) const {
      return (x == work_group.x) && (y == work_group.y) && (z == work_group.z);
    }
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
