#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Resource.h>

#define PRINT_FIELD(struct, field) #field << ": " << struct.field << std::endl

std::ostream& operator<<(std::ostream& out, VmaTotalStatistics stats) {
  VmaDetailedStatistics total_stats = stats.total;
  out << "VmaTotalStatistics: " << std::endl;
  out << "  " << PRINT_FIELD(total_stats.statistics, blockCount);
  out << "  " << PRINT_FIELD(total_stats.statistics, allocationCount);
  out << "  " << PRINT_FIELD(total_stats.statistics, blockBytes);
  out << "  " << PRINT_FIELD(total_stats.statistics, allocationBytes);
  return out;
}

#undef PRINT_FIELD

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// MemoryBarrier
//

MemoryBarrier::MemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags)
    : handle{
          VK_STRUCTURE_TYPE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
      } {}

//
// MemoryAllocation
//

MemoryAllocation::MemoryAllocation()
    : memory_requirements{},
      create_info{},
      allocator(VK_NULL_HANDLE),
      allocation(VK_NULL_HANDLE) {}

MemoryAllocation::MemoryAllocation(
    VmaAllocator vma_allocator,
    const VkMemoryRequirements& mem_props,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(mem_props),
      create_info(create_info),
      allocator(vma_allocator),
      allocation(VK_NULL_HANDLE) {
  VK_CHECK(vmaAllocateMemory(
      allocator, &memory_requirements, &create_info, &allocation, nullptr));
}

MemoryAllocation::MemoryAllocation(MemoryAllocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation) {
  other.allocation = VK_NULL_HANDLE;
}

MemoryAllocation& MemoryAllocation::operator=(
    MemoryAllocation&& other) noexcept {
  VmaAllocation tmp_allocation = allocation;

  memory_requirements = other.memory_requirements;
  create_info = other.create_info;
  allocator = other.allocator;
  allocation = other.allocation;

  other.allocation = tmp_allocation;

  return *this;
}

MemoryAllocation::~MemoryAllocation() {
  if (VK_NULL_HANDLE != allocation) {
    vmaFreeMemory(allocator, allocation);
  }
}

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      handle_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_({
          size,
          0u,
          size,
          usage,
      }),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_(allocate_memory),
      handle_(VK_NULL_HANDLE) {
  // Only allocate memory if the buffer has non-zero size
  if (size == 0) {
    return;
  }

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // size
      buffer_properties_.buffer_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        nullptr));
  } else {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;
  bool tmp_owns_memory = owns_memory_;

  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  handle_ = other.handle_;

  other.handle_ = tmp_buffer;
  other.owns_memory_ = tmp_owns_memory;

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {
    if (owns_memory_) {
      vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyBuffer, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
  }
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);
  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_{buffer.mem_size()} {
  if (allocation_) {
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_{other.data_len_} {
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
}

MemoryMap::~MemoryMap() {
  if (!data_) {
    return;
  }

  if (allocation_) {
    if (access_ & MemoryAccessType::WRITE) {
      // Call will be ignored by implementation if the memory type this
      // allocation belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is
      // the behavior we want. Don't check the result here as the destructor
      // cannot throw.
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
    }

    vmaUnmapMemory(allocator_, allocation_);
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle_, // buffer
          buffer.buffer_properties_.mem_offset, // offset
          buffer.buffer_properties_.mem_range, // size
      } {}

//
// ImageSampler
//

bool operator==(
    const ImageSampler::Properties& _1,
    const ImageSampler::Properties& _2) {
  return (
      _1.filter == _2.filter && _1.mipmap_mode == _2.mipmap_mode &&
      _1.address_mode == _2.address_mode && _1.border_color == _2.border_color);
}

ImageSampler::ImageSampler(
    VkDevice device,
    const ImageSampler::Properties& props)
    : device_(device), handle_(VK_NULL_HANDLE) {
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      props.filter, // magFilter
      props.filter, // minFilter
      props.mipmap_mode, // mipmapMode
      props.address_mode, // addressModeU
      props.address_mode, // addressModeV
      props.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      1.0f, // maxAnisotropy,
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      VK_LOD_CLAMP_NONE, // maxLod
      props.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  VK_CHECK(vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
}

ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ImageSampler::~ImageSampler() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroySampler(device_, handle_, nullptr);
}

size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  size_t seed = 0;
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(props.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(props.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(props.address_mode));
  seed =
      utils::hash_combine(seed, std::hash<VkBorderColor>()(props.border_color));
  return seed;
}

void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkSampler tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// VulkanImage
//

VulkanImage::VulkanImage()
    : image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{} {}

VulkanImage::VulkanImage(
    VmaAllocator vma_allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    const VkImageLayout layout,
    VkSampler sampler,
    const bool allocate_memory)
    : image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_{allocate_memory},
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout) {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  // If any dims are zero, then no memory will be allocated for the image.
  if (image_props.image_extents.width == 0 ||
      image_props.image_extents.height == 0 ||
      image_props.image_extents.depth == 0) {
    return;
  }

  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      image_properties_.image_type, // imageType
      image_properties_.image_format, // format
      image_properties_.image_extents, // extents
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      VK_IMAGE_TILING_OPTIMAL, // tiling
      image_properties_.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout_, // initialLayout
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    VK_CHECK(vmaCreateImage(
        allocator_,
        &image_create_info,
        &allocation_create_info,
        &(handles_.image),
        &(memory_.allocation),
        nullptr));
    // Only create the image view if the image has been bound to memory
    create_image_view();
  } else {
    VK_CHECK(vkCreateImage(
        allocator_info.device, &image_create_info, nullptr, &(handles_.image)));
  }
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handles_(other.handles_),
      layout_(other.layout_) {
  other.handles_.image = VK_NULL_HANDLE;
  other.handles_.image_view = VK_NULL_HANDLE;
  other.handles_.sampler = VK_NULL_HANDLE;
  other.owns_memory_ = false;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  VkImage tmp_image = handles_.image;
  VkImageView tmp_image_view = handles_.image_view;
  bool tmp_owns_memory = owns_memory_;

  image_properties_ = other.image_properties_;
  view_properties_ = other.view_properties_;
  sampler_properties_ = other.sampler_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  handles_ = other.handles_;
  layout_ = other.layout_;

  other.handles_.image = tmp_image;
  other.handles_.image_view = tmp_image_view;
  other.owns_memory_ = tmp_owns_memory;

  return *this;
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE != handles_.image_view) {
    vkDestroyImageView(this->device(), handles_.image_view, nullptr);
  }

  if (VK_NULL_HANDLE != handles_.image) {
    if (owns_memory_) {
      vmaDestroyImage(allocator_, handles_.image, memory_.allocation);
    } else {
      vkDestroyImage(this->device(), handles_.image, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyImage, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
  }
}

void VulkanImage::create_image_view() {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  const VkComponentMapping component_mapping{
      VK_COMPONENT_SWIZZLE_IDENTITY, // r
      VK_COMPONENT_SWIZZLE_IDENTITY, // g
      VK_COMPONENT_SWIZZLE_IDENTITY, // b
      VK_COMPONENT_SWIZZLE_IDENTITY, // a
  };

  const VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // baseMipLevel
      VK_REMAINING_MIP_LEVELS, // levelCount
      0u, // baseArrayLayer
      VK_REMAINING_ARRAY_LAYERS, // layerCount
  };

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      handles_.image, // image
      view_properties_.view_type, // viewType
      view_properties_.view_format, // format
      component_mapping, // components
      subresource_range, // subresourceRange
  };

  VK_CHECK(vkCreateImageView(
      allocator_info.device,
      &(image_view_create_info),
      nullptr,
      &(handles_.image_view)));
}

VkMemoryRequirements VulkanImage::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(
      this->device(), handles_.image, &memory_requirements);
  return memory_requirements;
}

//
// ImageMemoryBarrier
//

ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          src_layout_flags, // oldLayout
          dst_layout_flags, // newLayout
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          image.handles_.image, // image
          {
              // subresourceRange
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
              0u, // baseMipLevel
              VK_REMAINING_MIP_LEVELS, // levelCount
              0u, // baseArrayLayer
              VK_REMAINING_ARRAY_LAYERS, // layerCount
          },
      } {}

//
// SamplerCache
//

SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

SamplerCache::~SamplerCache() {
  purge();
}

VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, SamplerCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void SamplerCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// MemoryAllocator
//

MemoryAllocator::MemoryAllocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_{},
      physical_device_(physical_device),
      device_(device),
      allocator_{VK_NULL_HANDLE} {
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags
      physical_device_, // physicalDevice
      device_, // device
      0u, // preferredLargeHeapBlockSize
      nullptr, // pAllocationCallbacks
      nullptr, // pDeviceMemoryCallbacks
      nullptr, // pHeapSizeLimit
      &vk_functions, // pVulkanFunctions
      instance, // instance
      VK_API_VERSION_1_0, // vulkanApiVersion
      nullptr, // pTypeExternalMemoryHandleTypes
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_));
}

MemoryAllocator::MemoryAllocator(MemoryAllocator&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      allocator_(other.allocator_) {
  other.allocator_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.instance_ = VK_NULL_HANDLE;
}

MemoryAllocator::~MemoryAllocator() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }
  vmaDestroyAllocator(allocator_);
}

MemoryAllocation MemoryAllocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  VmaAllocationCreateInfo alloc_create_info = create_info;
  // Protect against using VMA_MEMORY_USAGE_AUTO_* flags when allocating memory
  // directly, since those usage flags require that VkBufferCreateInfo and/or
  // VkImageCreateInfo also be available.
  switch (create_info.usage) {
    // The logic for the below usage options are too complex, therefore prevent
    // those from being used with direct memory allocation.
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
      VK_THROW(
          "Only the VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE usage flag is compatible with create_allocation()");
      break;
    // Most of the time, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE will simply set the
    // DEVICE_LOCAL_BIT as a preferred memory flag. Therefore the below is a
    // decent approximation for VMA behaviour.
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
      alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      break;
    default:
      break;
  }

  return MemoryAllocation(allocator_, memory_requirements, alloc_create_info);
}

VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  const VulkanImage::ImageProperties image_props{
      image_type,
      image_format,
      extents,
      usage,
  };

  const VulkanImage::ViewProperties view_props{
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  return VulkanImage(
      allocator_,
      alloc_create_info,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler,
      allocate_memory);
}

VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only,
    const bool allocate_memory) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // The create storage buffer will be accessed by both the CPU and GPU, so set
  // the appropriate flags to indicate that the host device will be accessing
  // the data from this buffer.
  if (!gpu_only) {
    // Deferred memory allocation should only be used for GPU only buffers.
    VK_CHECK_COND(
        allocate_memory,
        "Only GPU-only buffers should use deferred memory allocation");

    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  return VulkanBuffer(
      allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
}

VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

  VkBufferUsageFlags buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  VulkanBuffer uniform_buffer(
      allocator_, size, alloc_create_info, buffer_usage);
  return uniform_buffer;
}

//
// VulkanFence
//

VulkanFence::VulkanFence()
    : device_(VK_NULL_HANDLE), handle_(VK_NULL_HANDLE), waiting_(false) {}

VulkanFence::VulkanFence(VkDevice device)
    : device_(device), handle_(VK_NULL_HANDLE), waiting_(VK_NULL_HANDLE) {
  const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
  };

  VK_CHECK(vkCreateFence(device_, &fence_create_info, nullptr, &handle_));
}

VulkanFence::VulkanFence(VulkanFence&& other) noexcept
    : device_(other.device_), handle_(other.handle_), waiting_(other.waiting_) {
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;
}

VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  waiting_ = other.waiting_;

  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;

  return *this;
}

VulkanFence::~VulkanFence() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyFence(device_, handle_, nullptr);
}

void VulkanFence::wait() {
  // if get_submit_handle() has not been called, then this will no-op
  if (waiting_) {
    VkResult fence_status = VK_NOT_READY;
    // Run the wait in a loop to keep the CPU hot. A single call to
    // vkWaitForFences with no timeout may cause the calling thread to be
    // scheduled out.
    do {
      // The timeout (last) arg is in units of ns
      fence_status = vkWaitForFences(device_, 1u, &handle_, VK_TRUE, 100000);

      VK_CHECK_COND(
          fence_status != VK_ERROR_DEVICE_LOST,
          "Vulkan Fence: Device lost while waiting for fence!");
    } while (fence_status != VK_SUCCESS);

    VK_CHECK(vkResetFences(device_, 1u, &handle_));

    waiting_ = false;
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
