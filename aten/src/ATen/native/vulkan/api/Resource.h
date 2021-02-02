#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Resource final {
  class Pool;

  //
  // Memory
  //

  struct Memory final {
    /*
      Descriptor
    */

    struct Descriptor final {
      VmaMemoryUsage usage;
      VkMemoryPropertyFlags /* optional */ required;
      VkMemoryPropertyFlags /* optional */ preferred;
    };

    /*
      Barrier
    */

    struct Barrier final {
      VkAccessFlags src;
      VkAccessFlags dst;
    };

    /*
      Access
    */

    struct Access final {
      typedef uint8_t Flags;

      enum Type : Flags {
        None = 0u << 0u,
        Read = 1u << 0u,
        Write = 1u << 1u,
      };

      template<typename Type, Flags access>
      using Pointer = std::add_pointer_t<
          std::conditional_t<
              0u != (access & Write),
              Type,
              std::add_const_t<Type>>>;
    };

    class Scope;
    template<typename Type>
    using Handle = Handle<Type, Scope>;

    template<
        typename Type,
        typename Pointer = Access::Pointer<Type, Access::Read>>
    Handle<Pointer> map() const &;

    template<
        typename Type,
        Access::Flags kAccess,
        typename Pointer = Access::Pointer<Type, kAccess>>
    Handle<Pointer> map() &;

    VmaAllocator allocator;
    VmaAllocation allocation;

   private:
    // Intentionally disabed to ensure memory access is always properly
    // encapsualted in a scoped map-unmap region.  Allowing below overloads
    // to be invoked on a temporary would open the door to the possibility
    // of accessing the underlying memory out of the expected scope making
    // for seemingly ineffective memory writes and hard to hunt down bugs.

    template<typename Type, typename Pointer>
    Handle<Pointer> map() const && = delete;

    template<typename Type, Access::Flags kAccess, typename Pointer>
    Handle<Pointer> map() && = delete;
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
        Memory::Descriptor memory;
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

  typedef enum VkImagePackFormat {
    VK_IMAGE_PACK_NC4HW_3D = 0,
    VK_IMAGE_PACK_NC4HW_2D = 1,
    VK_IMAGE_PACK_H2W2 = 2,
  } VkImagePackFormat;

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
        Memory::Descriptor memory;
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
  // Fence
  //

  struct Fence final {
    Pool* pool;
    size_t id;

    operator bool() const;
    VkFence handle(bool add_to_waitlist = true) const;
    void wait(uint64_t timeout_nanoseconds = UINT64_MAX);
  };

  //
  // Pool
  //

  class Pool final {
   public:
    class Policy {
     public:
      virtual ~Policy() = default;

      static std::unique_ptr<Policy> linear(
          VkDeviceSize block_size = VMA_DEFAULT_LARGE_HEAP_BLOCK_SIZE,
          uint32_t min_block_count = 1u,
          uint32_t max_block_count = UINT32_MAX);

      virtual void enact(
          VmaAllocator allocator,
          const VkMemoryRequirements& memory_requirements,
          VmaAllocationCreateInfo& allocation_create_info) = 0;
    };

    explicit Pool(const GPU& gpu, std::unique_ptr<Policy> = {});
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&);
    Pool& operator=(Pool&&);
    ~Pool();

    // Primary

    Buffer buffer(const Buffer::Descriptor& descriptor);
    Image image(const Image::Descriptor& descriptor);
    Fence fence();
    void purge();

    // Helper

    template <typename Block>
    Buffer uniform(const Block& block);

   private:
    friend struct Fence;

    void invalidate();

   private:
    struct Configuration final {
      static constexpr uint32_t kReserve = 256u;
    };

    VkDevice device_;
    Handle<VmaAllocator, void(*)(VmaAllocator)> allocator_;

    struct {
      std::unique_ptr<Policy> policy;
    } memory_;

    struct {
      std::vector<Handle<Buffer, void(*)(const Buffer&)>> pool;
    } buffer_;

    struct {
      std::vector<Handle<Image, void(*)(const Image&)>> pool;
      Image::Sampler sampler;
    } image_;

    struct {
      std::vector<Handle<VkFence, VK_DELETER(Fence)>> pool;
      mutable std::vector<VkFence> waitlist;
      size_t in_use;
    } fence_;
  } pool;

  explicit Resource(const GPU& gpu)
    : pool(gpu, Pool::Policy::linear()) {
  }
};

//
// Impl
//

class Resource::Memory::Scope final {
 public:
  Scope(
      VmaAllocator allocator,
      VmaAllocation allocation,
      Access::Flags access);

  void operator()(const void* data) const;

 private:
  VmaAllocator allocator_;
  VmaAllocation allocation_;
  Access::Flags access_;
};

template<typename, typename Pointer>
inline Resource::Memory::Handle<Pointer> Resource::Memory::map() const & {
  // Forward declaration
  void* map(const Memory&, Access::Flags);

  return Handle<Pointer>{
    reinterpret_cast<Pointer>(map(*this, Access::Read)),
    Scope(allocator, allocation, Access::Read),
  };
}

template<typename, Resource::Memory::Access::Flags kAccess, typename Pointer>
inline Resource::Memory::Handle<Pointer> Resource::Memory::map() & {
  // Forward declaration
  void* map(const Memory&, Access::Flags);

  static_assert(
      (kAccess == Access::Read) ||
      (kAccess == Access::Write) ||
      (kAccess == (Access::Read | Access::Write)),
      "Invalid memory access!");

  return Handle<Pointer>{
    reinterpret_cast<Pointer>(map(*this, kAccess)),
    Scope(allocator, allocation, kAccess),
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
    static_assert(
      std::is_trivially_copyable<Resource::Image::Sampler::Descriptor>::value,
      "This implementation is no longer valid!");

  return (0 == memcmp(&_1, &_2, sizeof(Resource::Image::Sampler::Descriptor)));
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

inline Resource::Fence::operator bool() const {
  return pool;
}

template<typename Block>
inline Resource::Buffer Resource::Pool::uniform(const Block& block) {
  Buffer uniform = this->buffer({
      sizeof(Block),
      {
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        {
          VMA_MEMORY_USAGE_CPU_TO_GPU,
          0u,
          0u,
        },
      },
    });

  {
    Memory::Handle<Block*> memory = uniform.memory.template map<
        Block,
        Memory::Access::Write>();

    *memory.get() = block;
  }

  return uniform;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
