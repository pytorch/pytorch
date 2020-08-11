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

    enum class Access;
    class Scope;

    template<typename Type>
    using Data = Handle<Type, Scope>;

    template<typename Type, typename ConstType = std::add_const_t<Type>>
    Data<ConstType> Map() const;

    template<typename Type>
    Data<Type> Map();
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
        VkImageUsageFlags image;
        VmaMemoryUsage memory;
      } usage;

      struct {
        VkImageViewType type;
        VkFormat format;
      } view;
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

//
// Impl
//

enum class Resource::Memory::Access {
  Read,
  Write,
};

class Resource::Memory::Scope final {
 public:
  Scope(VmaAllocator allocator, VmaAllocation allocation, Access access);
  void operator()(const void* data) const;

 private:
  VmaAllocator allocator_;
  VmaAllocation allocation_;
  Access access_;
};

template<typename Type, typename ConstType>
inline Resource::Memory::Data<ConstType> Resource::Memory::Map() const {
  Type data{};
  VK_CHECK(vmaMapMemory(allocator, allocation, data));

  return Data<ConstType>{
    data,
    Scope(allocator, allocation, Access::Read),
  };
}

template<typename Type>
inline Resource::Memory::Data<Type> Resource::Memory::Map() {
  Type data{};

  VK_CHECK(vmaMapMemory(allocator, allocation, data));

  return Data<Type>{
    data,
    Scope(allocator, allocation, Access::Write),
  };
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
