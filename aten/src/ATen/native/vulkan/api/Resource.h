#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Allocator.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <mutex>
#include <ostream>
#include <stack>
#include <unordered_map>

std::ostream& operator<<(std::ostream& out, VmaTotalStatistics stats);

namespace at {
namespace native {
namespace vulkan {
namespace api {

using MemoryAccessFlags = uint8_t;

constexpr VmaAllocationCreateFlags DEFAULT_ALLOCATION_STRATEGY =
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;

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

struct MemoryAllocation final {
  explicit MemoryAllocation();

  explicit MemoryAllocation(
      const VmaAllocator,
      const VkMemoryRequirements&,
      const VmaAllocationCreateInfo&);

  MemoryAllocation(const MemoryAllocation&) = delete;
  MemoryAllocation& operator=(const MemoryAllocation&) = delete;

  MemoryAllocation(MemoryAllocation&&) noexcept;
  MemoryAllocation& operator=(MemoryAllocation&&) noexcept;

  ~MemoryAllocation();

  VkMemoryRequirements memory_requirements;
  // The properties this allocation was created with
  VmaAllocationCreateInfo create_info;
  // The allocator object this was allocated from
  VmaAllocator allocator;
  // Handles to the allocated memory
  VmaAllocation allocation;

  operator bool() const {
    return (allocation != VK_NULL_HANDLE);
  }
};

class VulkanBuffer final {
 public:
  struct BufferProperties final {
    VkDeviceSize size;
    VkDeviceSize mem_offset;
    VkDeviceSize mem_range;
    VkBufferUsageFlags buffer_usage;
  };

  explicit VulkanBuffer();

  explicit VulkanBuffer(
      const VmaAllocator,
      const VkDeviceSize,
      const VmaAllocationCreateInfo&,
      const VkBufferUsageFlags,
      const bool allocate_memory = true);

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
  BufferProperties buffer_properties_;
  VmaAllocator allocator_;
  MemoryAllocation memory_;
  // Indicates whether the underlying memory is owned by this resource
  bool owns_memory_;
  VkBuffer handle_;

 public:
  inline VkDevice device() const {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    return allocator_info.device;
  }

  inline VmaAllocator vma_allocator() const {
    return allocator_;
  }

  inline VmaAllocation allocation() const {
    return memory_.allocation;
  }

  inline VmaAllocationCreateInfo allocation_create_info() const {
    return VmaAllocationCreateInfo(memory_.create_info);
  }

  inline VkBuffer handle() const {
    return handle_;
  }

  inline VkDeviceSize mem_offset() const {
    return buffer_properties_.mem_offset;
  }

  inline VkDeviceSize mem_range() const {
    return buffer_properties_.mem_range;
  }

  inline VkDeviceSize mem_size() const {
    return buffer_properties_.size;
  }

  inline bool has_memory() const {
    return (memory_.allocation != VK_NULL_HANDLE);
  }

  inline bool owns_memory() const {
    return owns_memory_;
  }

  operator bool() const {
    return (handle_ != VK_NULL_HANDLE);
  }

  inline void bind_allocation(const MemoryAllocation& memory) {
    VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
    VK_CHECK(vmaBindBufferMemory(allocator_, memory.allocation, handle_));
    memory_.allocation = memory.allocation;
  }

  VkMemoryRequirements get_memory_requirements() const;
};

class MemoryMap final {
 public:
  explicit MemoryMap(
      const VulkanBuffer& buffer,
      const MemoryAccessFlags access);

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
  VkDeviceSize data_len_;

 public:
  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(data_);
  }

  inline size_t nbytes() {
    return utils::safe_downcast<size_t>(data_len_);
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

  explicit ImageSampler(VkDevice, const Properties&);

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
  struct ImageProperties final {
    VkImageType image_type;
    VkFormat image_format;
    VkExtent3D image_extents;
    VkImageUsageFlags image_usage;
  };

  struct ViewProperties final {
    VkImageViewType view_type;
    VkFormat view_format;
  };

  using SamplerProperties = ImageSampler::Properties;

  struct Handles final {
    VkImage image;
    VkImageView image_view;
    VkSampler sampler;
  };

  explicit VulkanImage();

  explicit VulkanImage(
      const VmaAllocator,
      const VmaAllocationCreateInfo&,
      const ImageProperties&,
      const ViewProperties&,
      const SamplerProperties&,
      const VkImageLayout layout,
      VkSampler,
      const bool allocate_memory = true);

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
  ImageProperties image_properties_;
  ViewProperties view_properties_;
  SamplerProperties sampler_properties_;
  // The allocator object this was allocated from
  VmaAllocator allocator_;
  // Handles to the allocated memory
  MemoryAllocation memory_;
  // Indicates whether the underlying memory is owned by this resource
  bool owns_memory_;
  Handles handles_;
  // Layout
  VkImageLayout layout_;

 public:
  void create_image_view();

  inline VkDevice device() const {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    return allocator_info.device;
  }

  inline VmaAllocator vma_allocator() const {
    return allocator_;
  }

  inline VmaAllocation allocation() const {
    return memory_.allocation;
  }

  inline VmaAllocationCreateInfo allocation_create_info() const {
    return VmaAllocationCreateInfo(memory_.create_info);
  }

  inline VkFormat format() const {
    return image_properties_.image_format;
  }

  inline VkExtent3D extents() const {
    return image_properties_.image_extents;
  }

  inline VkImage handle() const {
    return handles_.image;
  }

  inline VkImageView image_view() const {
    return handles_.image_view;
  }

  inline VkSampler sampler() const {
    return handles_.sampler;
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

  inline bool has_memory() const {
    return (memory_.allocation != VK_NULL_HANDLE);
  }

  inline bool owns_memory() const {
    return owns_memory_;
  }

  inline operator bool() const {
    return (handles_.image != VK_NULL_HANDLE);
  }

  inline void bind_allocation(const MemoryAllocation& memory) {
    VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
    VK_CHECK(vmaBindImageMemory(allocator_, memory.allocation, handles_.image));
    memory_.allocation = memory.allocation;

    // Only create the image view if the image has been bound to memory
    create_image_view();
  }

  VkMemoryRequirements get_memory_requirements() const;
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
  explicit SamplerCache(VkDevice device);

  SamplerCache(const SamplerCache&) = delete;
  SamplerCache& operator=(const SamplerCache&) = delete;

  SamplerCache(SamplerCache&&) noexcept;
  SamplerCache& operator=(SamplerCache&&) = delete;

  ~SamplerCache();

  using Key = ImageSampler::Properties;
  using Value = ImageSampler;
  using Hasher = ImageSampler::Hasher;

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkSampler retrieve(const Key&);
  void purge();
};

class MemoryAllocator final {
 public:
  explicit MemoryAllocator(
      VkInstance instance,
      VkPhysicalDevice physical_device,
      VkDevice device);

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
  MemoryAllocation create_allocation(
      const VkMemoryRequirements& memory_requirements,
      const VmaAllocationCreateInfo& create_info);

  VulkanImage create_image(
      const VkExtent3D&,
      const VkFormat,
      const VkImageType,
      const VkImageViewType,
      const VulkanImage::SamplerProperties&,
      VkSampler,
      const bool allow_transfer = false,
      const bool allocate_memory = true);

  VulkanBuffer create_storage_buffer(
      const VkDeviceSize,
      const bool gpu_only = true,
      const bool allocate_memory = true);

  VulkanBuffer create_staging_buffer(const VkDeviceSize);

  /*
   * Create a uniform buffer with a specified size
   */
  VulkanBuffer create_uniform_buffer(const VkDeviceSize);

  /*
   * Create a uniform buffer containing the data in an arbitrary struct
   */
  template <typename Block>
  VulkanBuffer create_params_buffer(const Block& block);

  VmaTotalStatistics get_memory_statistics() const {
    VmaTotalStatistics stats = {};
    vmaCalculateStatistics(allocator_, &stats);
    return stats;
  }
};

class VulkanFence final {
 public:
  // TODO: This is required for the lazy allocation pattern in api/Tensor.
  //       It will be disabled pending future refactors.
  explicit VulkanFence();

  explicit VulkanFence(VkDevice);

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

  explicit FencePool(VkDevice device) : device_(device), pool_{} {}

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

//
// Impl
//

template <typename Block>
inline VulkanBuffer MemoryAllocator::create_params_buffer(const Block& block) {
  VulkanBuffer uniform_buffer = create_uniform_buffer(sizeof(Block));

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
