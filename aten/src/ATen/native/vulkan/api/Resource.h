#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <ATen/native/vulkan/api/Command.h>
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

  class Memory final {
   public:
    constexpr Memory();

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
    friend class Pool;
    Memory(VmaAllocator, VmaAllocation);
    void* map() const;

    // Intentionally disabed to ensure memory access is always properly
    // encapsualted in a scoped map-unmap region.  Allowing below overloads
    // to be invoked on a temporary would open the door to the possibility
    // of accessing the underlying memory out of the expected scope making
    // for seemingly ineffective memory writes and hard to hunt down bugs.

    template<typename Type, typename Pointer>
    Data<Pointer> map() const && = delete;

    template<typename Type, typename Pointer>
    Data<Pointer> map() && = delete;

   private:
    VmaAllocator allocator_;
    VmaAllocation allocation_;
  };

  //
  // Buffer
  //

  class Buffer final {
   public:
    constexpr Buffer();

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

    class Object final {
     public:
      constexpr Object();

      operator bool() const;
      VkBuffer handle() const;
      VkDeviceSize offset() const;
      VkDeviceSize range() const;

     private:
      friend class Pool;
      Object(VkBuffer, VkDeviceSize, VkDeviceSize);

     private:
      VkBuffer handle_;
      VkDeviceSize offset_;
      VkDeviceSize range_;
    };

    operator bool() const;

    const Object& object() const;
    Object& object();

    const Memory& memory() const;
    Memory& memory();

   private:
    friend class Pool;
    Buffer(const Object&, const Memory&);

   private:
    Object object_;
    Memory memory_;
  };

  //
  // Image
  //

  class Image final {
   public:
    constexpr Image();

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

    class Object final {
     public:
      constexpr Object();

      operator bool() const;
      VkImage handle() const;
      VkImageLayout layout() const;
      VkImageView view() const;
      VkSampler sampler() const;

      void transition(
          const Command::Buffer& command_buffer,
          VkImageLayout image_layout);

     private:
      friend class Pool;
      Object(VkImage, VkImageLayout, VkImageView, VkSampler);

     private:
      VkImage handle_;
      VkImageLayout layout_;
      VkImageView view_;
      VkSampler sampler_;
    };

    operator bool() const;

    const Object& object() const;
    Object& object();

    const Memory& memory() const;
    Memory& memory();

   private:
    friend class Pool;
    Image(const Object&, const Memory&);

   private:
    Object object_;
    Memory memory_;
  };

  //
  // Fence
  //

  class Fence final {
   public:
    constexpr Fence();

    operator bool() const;
    VkFence handle() const;
    void wait(uint64_t timeout_nanoseconds = UINT64_MAX);

   private:
    friend class Pool;
    Fence(VkDevice, VkFence);

   private:
    VkDevice device_;
    VkFence handle_;
    mutable bool used_;
  };

  //
  // Pool
  //

  class Pool final {
   public:
    explicit Pool(const GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;
    ~Pool() = default;

    Buffer buffer(const Buffer::Descriptor& descriptor);
    Image image(const Image::Descriptor& descriptor);
    Fence fence();
    void purge();

   private:
    static void release_buffer(const Resource::Buffer&);
    static void release_image(const Resource::Image&);
    static void release_fence(Resource::Fence&);

   private:
    struct Configuration final {
      static constexpr uint32_t kReserve = 256u;
    };

    VkDevice device_;
    Handle<VmaAllocator, void(*)(VmaAllocator)> allocator_;

    struct {
      std::vector<Handle<Buffer, void(*)(const Buffer&)>> pool;
    } buffer_;

    struct {
      std::vector<Handle<Image, void(*)(const Image&)>> pool;
      Image::Sampler sampler;
    } image_;

    struct {
      std::vector<Handle<Fence, void(*)(Fence&)>> pool;
      std::vector<VkFence> free;
      std::vector<VkFence> used;
    } fence_;
  } pool;

  explicit Resource(const GPU& gpu)
    : pool(gpu) {
  }
};

//
// Impl
//

inline constexpr Resource::Memory::Memory()
  : allocator_{},
    allocation_{} {
}

inline Resource::Memory::Memory(
  const VmaAllocator allocator,
  const VmaAllocation allocation)
  : allocator_(allocator),
    allocation_(allocation) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator_,
      "Invalid VMA allocator!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocation_,
      "Invalid VMA allocation!");
}

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
  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator_, allocation_, Scope::Access::Read),
  };
}

template<typename, typename Pointer>
inline Resource::Memory::Data<Pointer> Resource::Memory::map() & {
  return Data<Pointer>{
    reinterpret_cast<Pointer>(map(*this)),
    Scope(allocator_, allocation_, Scope::Access::Write),
  };
}

inline constexpr Resource::Buffer::Object::Object()
  : handle_{},
    offset_{},
    range_{} {
}

inline Resource::Buffer::Object::Object(
    const VkBuffer buffer,
    const VkDeviceSize offset,
    const VkDeviceSize range)
  : handle_(buffer),
    offset_(offset),
    range_(range) {
}

inline Resource::Buffer::Object::operator bool() const {
  return VK_NULL_HANDLE != handle();
}

inline VkBuffer Resource::Buffer::Object::handle() const {
  return handle_;
}

inline VkDeviceSize Resource::Buffer::Object::offset() const {
  return offset_;
}

inline VkDeviceSize Resource::Buffer::Object::range() const {
  return range_;
}

inline constexpr Resource::Buffer::Buffer()
  : object_{},
    memory_{} {
}

inline Resource::Buffer::Buffer(
    const Object& object,
    const Memory& memory)
  : object_(object),
    memory_(memory) {
}

inline Resource::Buffer::operator bool() const {
  return object();
}

inline const Resource::Buffer::Object& Resource::Buffer::object() const {
  return object_;
}

inline Resource::Buffer::Object& Resource::Buffer::object() {
  return object_;
}

inline const Resource::Memory& Resource::Buffer::memory() const {
  return memory_;
}

inline Resource::Memory& Resource::Buffer::memory() {
  return memory_;
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

inline constexpr Resource::Image::Object::Object()
  : handle_{},
    layout_{},
    view_{},
    sampler_{} {
}

inline Resource::Image::Object::Object(
    const VkImage image,
    const VkImageLayout layout,
    const VkImageView view,
    const VkSampler sampler)
  : handle_(image),
    layout_(layout),
    view_(view),
    sampler_(sampler) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      handle_,
      "Invalid Vulkan image!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      view_,
      "Invalid Vulkan image view!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      sampler_,
      "Invalid Vulkan image sampler!");
}

inline Resource::Image::Object::operator bool() const {
  return VK_NULL_HANDLE != handle();
}

inline VkImage Resource::Image::Object::handle() const {
  return handle_;
}

inline VkImageLayout Resource::Image::Object::layout() const {
  return layout_;
}

inline VkImageView Resource::Image::Object::view() const {
  return view_;
}

inline VkSampler Resource::Image::Object::sampler() const {
  return sampler_;
}

inline constexpr Resource::Image::Image()
  : object_{},
    memory_{} {
}

inline Resource::Image::Image(
    const Object& object,
    const Memory& memory)
  : object_(object),
    memory_(memory) {
}

inline Resource::Image::operator bool() const {
  return object();
}

inline const Resource::Image::Object& Resource::Image::object() const {
  return object_;
}

inline Resource::Image::Object& Resource::Image::object() {
  return object_;
}

inline const Resource::Memory& Resource::Image::memory() const {
  return memory_;
}

inline Resource::Memory& Resource::Image::memory() {
  return memory_;
}

inline constexpr Resource::Fence::Fence()
  : device_{},
    handle_{},
    used_{false} {
}

inline Resource::Fence::Fence(
    const VkDevice device,
    const VkFence fence)
  : device_(device),
    handle_(fence),
    used_(false) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      handle_,
      "Invalid Vulkan fence!");
}

inline Resource::Fence::operator bool() const {
  return handle_ != VK_NULL_HANDLE;
}

inline VkFence Resource::Fence::handle() const {
  used_ = true;
  return handle_;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
