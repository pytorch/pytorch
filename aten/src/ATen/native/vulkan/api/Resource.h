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
    VkBuffer handle;
    Memory memory;

    struct Descriptor final {
      VkDeviceSize size;
      VkBufferUsageFlags usage;
    };
  };

  /*
    Image
  */

  struct Image final {
    VkImage handle;
    VkImageView view;
    Memory memory;

    struct Descriptor final {
      VkExtent3D extent;
    };
  };

  /*
    Pool
  */

  class Pool final {
   public:
    explicit Pool(VkDevice device);

    Buffer allocate(const Buffer::Descriptor& descriptor);
    Image allocate(const Image::Descriptor& descriptor);
    void purge();

   private:
    Handle<VmaAllocator, decltype(&vmaDestroyAllocator)> allocator_;
    std::vector<Handle<Buffer>> buffers_;
    std::vector<Handle<Image>> images_;
  };
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
