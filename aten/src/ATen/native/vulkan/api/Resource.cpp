#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Resource.h>

#include <c10/core/ScalarTypeToTypeMeta.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Utility Functions
//

/*
 * This function is used to determine what image format to use for a given
 * dtype.
 *
 * TODO: enable proper format selection between kFloat and kHalf.
 *
 * Context: due to limitations of the shader compilation system, at the moment
 * it is not possible to support both 32 bit and 16 bit float formats since
 * shaders will have to specify the format qualifier of texture inputs. Right
 * now, shaders are compiled with either rgba16f or rgba32f qualifiers depending
 * on whether USE_VULKAN_FP16_INFERENCE is set. Therefore, textures must be
 * always created with the corresponding VkFormat. Consequently, kHalf tensors
 * are currently unsupported in favor of enforcing inputs to be of kFloat dtype.
 */
VkFormat vk_format(const at::ScalarType dtype) {
  switch (dtype) {
    case kFloat:
#ifdef USE_VULKAN_FP16_INFERENCE
      return VK_FORMAT_R16G16B16A16_SFLOAT;
#else
      return VK_FORMAT_R32G32B32A32_SFLOAT;
#endif /* USE_VULKAN_FP16_INFERENCE */
    case c10::kQUInt8:
      return VK_FORMAT_R8G8B8A8_UINT;
    case c10::kQInt8:
      return VK_FORMAT_R8G8B8A8_SINT;
    case c10::kQInt32:
      return VK_FORMAT_R32G32B32A32_SINT;

    default:
      TORCH_CHECK(
          false, "Vulkan vk_format(): no corresponding format for dtype");
  }
}

/*
 * This function is used to map a texture format to a corresponding
 * c10::ScalarType. It is primarily used to set the data type of a
 * StorageBuffer object that will receive copied data from a texture.
 */
c10::ScalarType c10_scalartype(const VkFormat image_format) {
  switch (image_format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return c10::kFloat;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return c10::kHalf;
    case VK_FORMAT_R8G8B8A8_UINT:
      return c10::kQUInt8;

    default:
      TORCH_CHECK(false, "vulkan c10_scalartype(): Unknown VkFormat.");
  }
}

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
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : memory_properties_{},
      buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      allocation_(VK_NULL_HANDLE),
      handle_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    const VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VulkanBuffer::MemoryProperties& mem_props)
    : memory_properties_(mem_props),
      buffer_properties_({
          size,
          0u,
          size,
      }),
      allocator_(vma_allocator),
      allocation_(VK_NULL_HANDLE),
      handle_(VK_NULL_HANDLE) {
  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // size
      memory_properties_.buffer_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  // TODO: enable creation with a custom pool
  VmaAllocationCreateInfo alloc_create_info{
      memory_properties_.create_flags, // flags
      memory_properties_.memory_usage, // usage
      memory_properties_.required_mem_flags, // requiredFlags
      memory_properties_.preferred_mem_flags, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
      0.5f, // priority
  };

  VK_CHECK(vmaCreateBuffer(
      allocator_,
      &buffer_create_info,
      &alloc_create_info,
      &handle_,
      &allocation_,
      nullptr));
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : memory_properties_(other.memory_properties_),
      buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      handle_(other.handle_) {
  other.allocation_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  const VmaAllocation tmp_allocation = allocation_;
  const VkBuffer tmp_buffer = handle_;

  memory_properties_ = other.memory_properties_;
  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  allocation_ = other.allocation_;
  handle_ = other.handle_;

  other.allocation_ = tmp_allocation;
  other.handle_ = tmp_buffer;

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {
    vmaDestroyBuffer(allocator_, handle_, allocation_);
  }
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
  VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
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
  if (C10_UNLIKELY(!data_)) {
    return;
  }

  if (access_ & MemoryAccessType::WRITE) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want. Don't check the result here as the destructor cannot throw.
    vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
  }

  vmaUnmapMemory(allocator_, allocation_);
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ) {
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
    const VkDevice device,
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
  if C10_LIKELY (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroySampler(device_, handle_, nullptr);
}

size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  return c10::get_hash(
      props.filter, props.mipmap_mode, props.address_mode, props.border_color);
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
    : memory_properties_{},
      image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      allocation_(VK_NULL_HANDLE),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{} {}

VulkanImage::VulkanImage(
    const VmaAllocator vma_allocator,
    const VkDevice device,
    const MemoryProperties& mem_props,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    const VkImageLayout layout,
    const VkSampler sampler)
    : memory_properties_(mem_props),
      image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      allocation_(VK_NULL_HANDLE),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout) {
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
      memory_properties_.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout_, // initialLayout
  };

  // TODO: enable creation with a custom pool
  const VmaAllocationCreateInfo alloc_create_info{
      memory_properties_.create_flags, // flags
      memory_properties_.memory_usage, // usage
      memory_properties_.required_mem_flags, // requiredFlags
      memory_properties_.preferred_mem_flags, // preferredFlags
      0u, // memoryTypeBits
      VK_NULL_HANDLE, // pool
      nullptr, // pUserData
      0.5f, // priority
  };

  VK_CHECK(vmaCreateImage(
      allocator_,
      &image_create_info,
      &alloc_create_info,
      &(handles_.image),
      &allocation_,
      nullptr));

  // Image View

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
      device, &image_view_create_info, nullptr, &(handles_.image_view)));
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : memory_properties_(other.memory_properties_),
      image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      handles_(other.handles_),
      layout_(other.layout_) {
  other.allocation_ = VK_NULL_HANDLE;
  other.handles_.image = VK_NULL_HANDLE;
  other.handles_.image_view = VK_NULL_HANDLE;
  other.handles_.sampler = VK_NULL_HANDLE;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  const VmaAllocation tmp_allocation = allocation_;
  const VkImage tmp_image = handles_.image;
  const VkImageView tmp_image_view = handles_.image_view;

  memory_properties_ = other.memory_properties_;
  image_properties_ = other.image_properties_;
  view_properties_ = other.view_properties_;
  sampler_properties_ = other.sampler_properties_;
  allocator_ = other.allocator_;
  allocation_ = other.allocation_;
  handles_ = other.handles_;
  layout_ = other.layout_;

  other.allocation_ = tmp_allocation;
  other.handles_.image = tmp_image;
  other.handles_.image_view = tmp_image_view;

  return *this;
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE != handles_.image_view) {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    vkDestroyImageView(allocator_info.device, handles_.image_view, nullptr);
  }

  if (VK_NULL_HANDLE != handles_.image) {
    vmaDestroyImage(allocator_, handles_.image, allocation_);
  }
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

SamplerCache::SamplerCache(const VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
  cache_ = std::move(other.cache_);
}

SamplerCache::~SamplerCache() {
  purge();
}

VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if C10_UNLIKELY (cache_.cend() == it) {
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
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice device)
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
  if C10_LIKELY (VK_NULL_HANDLE == allocator_) {
    return;
  }
  vmaDestroyAllocator(allocator_);
}

VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    const VkSampler sampler,
    const bool allow_transfer) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  const VulkanImage::MemoryProperties mem_props{
      DEFAULT_ALLOCATION_STRATEGY,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      0u,
      0u,
      usage,
  };

  const VulkanImage::ImageProperties image_props{
      image_type,
      image_format,
      extents,
  };

  const VulkanImage::ViewProperties view_props{
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  return VulkanImage(
      allocator_,
      device_,
      mem_props,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler);
}

VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateFlags create_flags = DEFAULT_ALLOCATION_STRATEGY;
  if (!gpu_only) {
    create_flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  }

  const VmaMemoryUsage vma_usage =
      gpu_only ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO;

  const VkMemoryPropertyFlags required_mem_props =
      gpu_only ? 0u : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

  const VkMemoryPropertyFlags preferred_mem_props = gpu_only
      ? 0u
      : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

  const VulkanBuffer::MemoryProperties mem_props{
      create_flags,
      vma_usage,
      required_mem_props,
      preferred_mem_props,
      buffer_usage,
  };

  return VulkanBuffer(allocator_, size, mem_props);
}

VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  const VulkanBuffer::MemoryProperties mem_props{
      DEFAULT_ALLOCATION_STRATEGY,
      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      0u,
      0u,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
  };

  return VulkanBuffer(allocator_, size, mem_props);
}

VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  const VulkanBuffer::MemoryProperties mem_props{
      DEFAULT_ALLOCATION_STRATEGY |
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      VMA_MEMORY_USAGE_AUTO,
      0u,
      0u,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
  };

  VulkanBuffer uniform_buffer(allocator_, size, mem_props);

  return uniform_buffer;
}

//
// VulkanFence
//

VulkanFence::VulkanFence()
    : device_(VK_NULL_HANDLE), handle_(VK_NULL_HANDLE), waiting_(false) {}

VulkanFence::VulkanFence(const VkDevice device)
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
  if C10_LIKELY (VK_NULL_HANDLE == handle_) {
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

      TORCH_CHECK(
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
