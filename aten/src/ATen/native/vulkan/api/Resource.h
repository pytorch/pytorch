#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Resource final {
  //
  // Memory
  //

  struct Memory final {
    /*
      Barrier
    */

    struct Barrier final {
      VkAccessFlags src;
      VkAccessFlags dst;
    };

    VmaAllocator allocator;
    VmaAllocation allocation;

    class Scope;
    template<typename Type>
    using Data = Handle<Type, Scope>;

    template<
        typename Type,
        typename Pointer = std::add_pointer_t<std::add_const_t<Type>>>
    Data<Pointer> map() const &;

    template<
        typename Type,
        typename Pointer = std::add_pointer_t<Type>>
    Data<Pointer> map() &;

   private:
    // Intentionally disabed to ensure memory access is always properly
    // encapsualted in a scoped map-unmap region.  Allowing below overloads
    // to be invoked on a temporary would open the door to the possibility
    // of accessing the underlying memory out of the expected scope making
    // for seemingly ineffective memory writes and hard to hunt down bugs.

    template<typename Type, typename Pointer>
    Data<Pointer> map() const && = delete;

    template<typename Type, typename Pointer>
    Data<Pointer> map() && = delete;
  };

  //
  // Buffer
  //

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

    /*
      Object
    */

    struct Object final {
      VkBuffer handle;
      VkDeviceSize offset;
      VkDeviceSize range;

      operator bool() const;
    };

    /*
      Barrier
    */

    struct Barrier final {
      Object object;
      Memory::Barrier memory;
    };

    Object object;
    Memory memory;

    operator bool() const;
  };

  //
  // Image
  //

  struct Image final {
    //
    // Sampler
    //

    struct Sampler final {
      /*
        Descriptor
      */

      struct Descriptor final {
        VkFilter filter;
        VkSamplerMipmapMode mipmap_mode;
        VkSamplerAddressMode address_mode;
        VkBorderColor border;
      };

      /*
        Factory
      */

      class Factory final {
       public:
        explicit Factory(const GPU& gpu);

        typedef Sampler::Descriptor Descriptor;
        typedef VK_DELETER(Sampler) Deleter;
        typedef Handle<VkSampler, Deleter> Handle;

        struct Hasher {
          size_t operator()(const Descriptor& descriptor) const;
        };

        Handle operator()(const Descriptor& descriptor) const;

       private:
        VkDevice device_;
      };

      /*
        Cache
      */

      typedef api::Cache<Factory> Cache;
      Cache cache;

      explicit Sampler(const GPU& gpu)
        : cache(Factory(gpu)) {
      }
    };

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

      Sampler::Descriptor sampler;
    };

    /*
      Object
    */

    struct Object final {
      VkImage handle;
      VkImageLayout layout;
      VkImageView view;
      VkSampler sampler;

      operator bool() const;
    };

    /*
      Barrier
    */

    struct Barrier final {
      Object object;
      Memory::Barrier memory;

      struct {
        VkImageLayout src;
        VkImageLayout dst;
      } layout;
    };

    Object object;
    Memory memory;

    operator bool() const;
  };

  //
  // Pool
  //

  class Pool final {
   public:
    explicit Pool(const GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&);
    Pool& operator=(Pool&&);
    ~Pool() = default;

    Buffer buffer(const Buffer::Descriptor& descriptor);
    Image image(const Image::Descriptor& descriptor);
    void purge();

   private:
    struct Configuration final {
      static constexpr uint32_t kReserve = 256u;
    };

    VkDevice device_;
    Handle<VmaAllocator, void(*)(VmaAllocator)> allocator_;
    std::vector<Handle<Buffer, void(*)(const Buffer&)>> buffers_;
    std::vector<Handle<Image, void(*)(const Image&)>> images_;
    Image::Sampler sampler_;
  } pool;

  explicit Resource(const GPU& gpu)
    : pool(gpu) {
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
inline Resource::Memory::Data<Pointer> Resource::Memory::map() const & {
  void* map(const Memory& memory);

  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator, allocation, Scope::Access::Read),
  };
}

template<typename, typename Pointer>
inline Resource::Memory::Data<Pointer> Resource::Memory::map() & {
  void* map(const Memory& memory);

  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator, allocation, Scope::Access::Write),
  };
}

inline Resource::Buffer::Object::operator bool() const {
  return VK_NULL_HANDLE != handle;
}

inline Resource::Buffer::operator bool() const {
  return object;
}

inline bool operator==(
    const Resource::Image::Sampler::Descriptor& _1,
    const Resource::Image::Sampler::Descriptor& _2) {
    return (_1.filter == _2.filter) &&
           (_1.mipmap_mode == _2.mipmap_mode) &&
           (_1.address_mode == _2.address_mode) &&
           (_1.border == _2.border);
}

inline size_t Resource::Image::Sampler::Factory::Hasher::operator()(
    const Descriptor& descriptor) const {
  return c10::get_hash(
      descriptor.filter,
      descriptor.mipmap_mode,
      descriptor.address_mode,
      descriptor.border);
}

inline Resource::Image::Object::operator bool() const {
  return VK_NULL_HANDLE != handle;
}

inline Resource::Image::operator bool() const {
  return object;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
