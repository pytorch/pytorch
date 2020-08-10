#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Allocator.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Resource final {
  /*
    Memory
  */

  struct Memory final {
    VmaAllocator allocator;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
  };

  /*
    Buffer
  */

  struct Buffer final {
    /*
      Descriptor
    */

    struct Descriptor final {
      VkDeviceSize size;

      struct {
        VkBufferUsageFlags buffer;
        VmaMemoryUsage memory;
      } usage;
    };

    VkBuffer handle;
    Memory memory;

    inline operator bool() const {
      return VK_NULL_HANDLE != handle;
    }
  };

  /*
    Image
  */

  struct Image final {
    /*
      Descriptor
    */

    struct Descriptor final {
      VkImageType type;
      VkFormat format;
      VkExtent3D extent;

      struct {
        VkImageViewType type;
        VkFormat format;
      } view;

      struct {
        VkImageUsageFlags image;
        VmaMemoryUsage memory;
      } usage;
    };

    VkImage handle;
    VkImageView view;
    Memory memory;

    inline operator bool() const {
      return VK_NULL_HANDLE != handle;
    }
  };

  /*
    Pool
  */

  class Pool final {
   public:
    Pool(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device);

    Buffer allocate(const Buffer::Descriptor& descriptor);
    Image allocate(const Image::Descriptor& descriptor);
    void purge();

   private:
    struct Configuration final {
      static constexpr uint32_t kReserve = 256u;
    };

    VkDevice device_;
    Handle<VmaAllocator, void(*)(VmaAllocator)> allocator_;
    std::vector<Handle<Buffer, void(*)(const Buffer&)>> buffers_;
    std::vector<Handle<Image, void(*)(const Image&)>> images_;
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
