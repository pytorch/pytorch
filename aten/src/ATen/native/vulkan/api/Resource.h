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

    class Scope;
    template<typename Type>
    using Data = Handle<Type, Scope>;

    template<
        typename Type,
        typename Pointer = std::add_pointer_t<std::add_const_t<Type>>>
    Data<Pointer> map() const;

    template<
        typename Type,
        typename Pointer = std::add_pointer_t<Type>>
    Data<Pointer> map();
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

    operator bool() const;
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

    operator bool() const;
  };

  /*
    Pool
  */

  class Pool final {
   public:
    Pool(
        VkInstance instance,
        VkPhysicalDevice physical_device,
        VkDevice device);

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
  } pool;

  Resource(
      const VkInstance instance,
      const VkPhysicalDevice physical_device,
      const VkDevice device)
    : pool(instance, physical_device, device) {
  }
};

//
// Impl
//

class Resource::Memory::Scope final {
 public:
  enum class Access {
    Read,
    Write,
  };

  Scope(VmaAllocator allocator, VmaAllocation allocation, Access access);
  void operator()(const void* data) const;

 private:
  VmaAllocator allocator_;
  VmaAllocation allocation_;
  Access access_;
};

template<typename, typename Pointer>
inline Resource::Memory::Data<Pointer> Resource::Memory::map() const {
  void* map(const Memory& memory);

  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator, allocation, Scope::Access::Read),
  };
}

template<typename, typename Pointer>
inline Resource::Memory::Data<Pointer> Resource::Memory::map() {
  void* map(const Memory& memory);

  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator, allocation, Scope::Access::Write),
  };
}

inline Resource::Buffer::operator bool() const {
  return VK_NULL_HANDLE != handle;
}

inline Resource::Image::operator bool() const {
  return VK_NULL_HANDLE != handle;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
