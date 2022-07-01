#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Cache.h>
#include <c10/util/hash.h>

#include <stack>

namespace at {
namespace native {
namespace vulkan {
namespace api {

typedef uint8_t MemoryAccessFlags;

enum MemoryAccessType : MemoryAccessFlags {
  NONE = 0u << 0u,
  READ = 1u << 0u,
  WRITE = 1u << 1u,
};

struct MemoryBarrier final {
  VkMemoryBarrier handle;

  MemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags);
};

class VulkanBuffer final {
 public:
  struct MemoryProperties final {
    VmaMemoryUsage memory_usage;
    VkMemoryPropertyFlags required_mem_flags;
    VkMemoryPropertyFlags preferred_mem_flags;

    VkBufferUsageFlags buffer_usage;
  };

  struct BufferProperties final {
    VkDeviceSize size;
    VkDeviceSize mem_offset;
    VkDeviceSize mem_range;
  };

  explicit VulkanBuffer();

  explicit VulkanBuffer(
      const VmaAllocator, const VkDeviceSize, const MemoryProperties&);

  VulkanBuffer(const VulkanBuffer&) = delete;
  VulkanBuffer& operator=(const VulkanBuffer&) = delete;

  VulkanBuffer(VulkanBuffer&&) noexcept;
  VulkanBuffer& operator=(VulkanBuffer&&) noexcept;

  ~VulkanBuffer();

  struct Package final {
    VkBuffer handle;
    VkDeviceSize buffer_offset;
    VkDeviceSize buffer_range;
  };

  friend struct BufferMemoryBarrier;

 private:
  MemoryProperties memory_properties_;
  BufferProperties buffer_properties_;
  // The allocator object this was allocated from
  VmaAllocator allocator_;
  // Handles to the allocated memory
  VmaAllocation allocation_;
  VkBuffer handle_;

 public:
  VmaAllocator vma_allocator() const {
    return allocator_;
  }

  VmaAllocation allocation() const {
    return allocation_;
  }

  Package package() const {
    return {
      handle_,
      buffer_properties_.mem_offset,
      buffer_properties_.mem_range
    };
  }

  operator bool() const {
    return (allocation_ != VK_NULL_HANDLE);
  }
};

class MemoryMap final {
 public:
  explicit MemoryMap(const VulkanBuffer& buffer, const MemoryAccessFlags access);

  MemoryMap(const MemoryMap&) = delete;
  MemoryMap& operator=(const MemoryMap&) = delete;

  MemoryMap(MemoryMap&&) noexcept;
  MemoryMap& operator=(MemoryMap&&) = delete;

  ~MemoryMap();

 private:
  uint8_t access_;
  VmaAllocator allocator_;
  VmaAllocation allocation_;
  void* data_;

 public:
  template<typename T>
  T* data() {
    return reinterpret_cast<T*>(data_);
  }

  void invalidate();
};

struct BufferMemoryBarrier final {
  VkBufferMemoryBarrier handle;

  BufferMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VulkanBuffer& buffer);
};

class ImageSampler final {
 public:
  struct Properties final {
    VkFilter filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode address_mode;
    VkBorderColor border_color;
  };

  explicit ImageSampler(const VkDevice, const Properties&);

  ImageSampler(const ImageSampler&) = delete;
  ImageSampler& operator=(const ImageSampler&) = delete;

  ImageSampler(ImageSampler&&) noexcept;
  ImageSampler& operator=(ImageSampler&&) = delete;

  ~ImageSampler();

 private:
  VkDevice device_;
  VkSampler handle_;

 public:
  VkSampler handle() const {
    return handle_;
  }

  struct Hasher {
    size_t operator()(const Properties&) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept;
};

class VulkanImage final {
 public:
  struct MemoryProperties final {
    VmaMemoryUsage memory_usage;
    VkMemoryPropertyFlags required_mem_flags;
    VkMemoryPropertyFlags preferred_mem_flags;

    VkImageUsageFlags image_usage;
  };

  struct ImageProperties final {
    VkImageType image_type;
    VkFormat image_format;
    VkExtent3D image_extents;
  };

  struct ViewProperties final {
    VkImageViewType view_type;
    VkFormat view_format;
  };

  typedef ImageSampler::Properties SamplerProperties;

  struct Handles final {
    VkImage image;
    VkImageView image_view;
    VkSampler sampler;
  };

  explicit VulkanImage();

  explicit VulkanImage(
      const VmaAllocator,
      const VkDevice,
      const MemoryProperties&,
      const ImageProperties&,
      const ViewProperties&,
      const SamplerProperties&,
      const VkImageLayout layout,
      const VkSampler);

  VulkanImage(const VulkanImage&) = delete;
  VulkanImage& operator=(const VulkanImage&) = delete;

  VulkanImage(VulkanImage&&) noexcept;
  VulkanImage& operator=(VulkanImage&&) noexcept;

  ~VulkanImage();

  struct Package final {
    VkImage handle;
    VkImageLayout image_layout;
    VkImageView image_view;
    VkSampler image_sampler;
  };

  friend struct ImageMemoryBarrier;

 private:
  MemoryProperties memory_properties_;
  ImageProperties image_properties_;
  ViewProperties view_properties_;
  SamplerProperties sampler_properties_;
  // The allocator object this was allocated from
  VmaAllocator allocator_;
  // Handles to the allocated memory
  VmaAllocation allocation_;
  Handles handles_;
  // Layout
  VkImageLayout layout_;

 public:
  inline VmaAllocator vma_allocator() const {
    return allocator_;
  }

  inline VmaAllocation allocation() const {
    return allocation_;
  }

  inline VkImage handle() const {
    return handles_.image;
  }

  Package package() const {
    return {
      handles_.image,
      layout_,
      handles_.image_view,
      handles_.sampler,
    };
  }

  inline VkImageLayout layout() const {
    return layout_;
  }

  inline void set_layout(const VkImageLayout layout) {
    layout_ = layout;
  }

  inline operator bool() const {
    return (allocation_ != VK_NULL_HANDLE);
  }
};

struct ImageMemoryBarrier final {
  VkImageMemoryBarrier handle;

  ImageMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VkImageLayout src_layout_flags,
      const VkImageLayout dst_layout_flags,
      const VulkanImage& image);
};

class SamplerCache final {
 public:
  explicit SamplerCache(const VkDevice device);

  SamplerCache(const SamplerCache&) = delete;
  SamplerCache& operator=(const SamplerCache&) = delete;

  SamplerCache(SamplerCache&&) noexcept;
  SamplerCache& operator=(SamplerCache&&) = delete;

  ~SamplerCache();

  typedef ImageSampler::Properties Key;
  typedef ImageSampler Value;
  typedef ImageSampler::Hasher Hasher;

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  ska::flat_hash_map<Key, Value, Hasher> cache_;

 public:
  VkSampler retrieve(const Key&);
  void purge();
};

class MemoryAllocator final {
 public:
  explicit MemoryAllocator(
      const VkInstance instance,
      const VkPhysicalDevice physical_device,
      const VkDevice device);

  MemoryAllocator(const MemoryAllocator&) = delete;
  MemoryAllocator& operator=(const MemoryAllocator&) = delete;

  MemoryAllocator(MemoryAllocator&&) noexcept;
  MemoryAllocator& operator=(MemoryAllocator&&) = delete;

  ~MemoryAllocator();

 private:
  VkInstance instance_;
  VkPhysicalDevice physical_device_;
  VkDevice device_;
  VmaAllocator allocator_;

 public:
  VulkanImage create_image3d_fp(
      const VkExtent3D&,
      const VulkanImage::SamplerProperties&,
      const VkSampler,
      const bool allow_transfer = false);

  VulkanBuffer create_storage_buffer(
      const VkDeviceSize, const bool gpu_only = true);

  VulkanBuffer create_staging_buffer(const VkDeviceSize);

  template<typename Block>
  VulkanBuffer create_params_buffer(const Block& block);
};

class VulkanFence final {
 public:
  // TODO: This is required for the lazy allocation pattern in api/Tensor.
  //       It will be disabled pending future refactors.
  explicit VulkanFence();

  explicit VulkanFence(const VkDevice);

  VulkanFence(const VulkanFence&) = delete;
  VulkanFence& operator=(const VulkanFence&) = delete;

  VulkanFence(VulkanFence&&) noexcept;
  VulkanFence& operator=(VulkanFence&&) noexcept;

  ~VulkanFence();

 private:
  VkDevice device_;
  VkFence handle_;
  bool waiting_;

 public:
  // Used to get the handle for a queue submission.
  VkFence get_submit_handle() {
    if (handle_ != VK_NULL_HANDLE) {
      // Indicate we are now waiting for this fence to be signaled
      waiting_ = true;
    }
    return handle_;
  }

  VkFence handle() {
    return handle_;
  }

  // Trigger a synchronous wait for the fence to be signaled
  void wait();

  bool waiting() const {
    return waiting_;
  }

  operator bool() const {
    return (VK_NULL_HANDLE != handle_);
  }
};

// A pool to track created Fences and reuse ones that are available.
// Only intended to be modified by one thread at a time.
struct FencePool final {
  VkDevice device_;

  std::stack<VulkanFence> pool_;

  explicit FencePool(const VkDevice device)
    : device_(device),
      pool_{} {
  }

  // Returns an rvalue reference to a fence, so that it can be moved
  inline VulkanFence get_fence() {
    if (pool_.empty()) {
      VulkanFence new_fence = VulkanFence(device_);
      return new_fence;
    }

    VulkanFence top_fence = std::move(pool_.top());
    pool_.pop();

    return top_fence;
  }

  // Marks the fence as available
  inline void return_fence(VulkanFence& fence) {
    pool_.push(std::move(fence));
  }
};

/* ---- Old Code ---- */

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
        typedef api::Handle<VkSampler, Deleter> Handle;

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

    Buffer create_buffer(const Buffer::Descriptor& descriptor);
    void register_buffer_cleanup(const Buffer& buffer);
    Image create_image(const Image::Descriptor& descriptor);
    void register_image_cleanup(const Image& image);

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
    : pool(gpu, nullptr) {
  }
};

void release_buffer(const Resource::Buffer& buffer);

void release_image(const Resource::Image& image);

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

  return (_1.filter == _2.filter && \
          _1.mipmap_mode == _2.mipmap_mode && \
          _1.address_mode == _2.address_mode && \
          _1.border == _2.border);
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
  Buffer uniform = this->create_buffer({
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
  this->register_buffer_cleanup(uniform);

  {
    Memory::Handle<Block*> memory = uniform.memory.template map<
        Block,
        Memory::Access::Write>();

    *memory.get() = block;
  }

  return uniform;
}

template<typename Block>
inline VulkanBuffer MemoryAllocator::create_params_buffer(const Block& block) {
  const VulkanBuffer::MemoryProperties mem_props{
    VMA_MEMORY_USAGE_CPU_TO_GPU,
    0u,
    0u,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
  };

  VulkanBuffer uniform_buffer(allocator_, sizeof(Block), mem_props);

  // Fill the uniform buffer with data in block
  {
    MemoryMap mapping(uniform_buffer, MemoryAccessType::WRITE);
    Block* data_ptr = mapping.template data<Block>();

    *data_ptr = block;
  }

  return uniform_buffer;
}


} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
