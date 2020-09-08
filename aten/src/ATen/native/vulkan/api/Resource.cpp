#include <ATen/native/vulkan/api/Resource.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VmaAllocator create_allocator(
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice device) {
  TORCH_INTERNAL_ASSERT(instance, "Invalid Vulkan instance!");
  TORCH_INTERNAL_ASSERT(physical_device, "Invalid Vulkan physical device!");
  TORCH_INTERNAL_ASSERT(device, "Invalid Vulkan device!");

  const VmaAllocatorCreateInfo allocator_create_info{
    0u,
    physical_device,
    device,
    0u,
    nullptr,
    nullptr,
    1u,
    nullptr,
    nullptr, // TODO (Ashkan): VULKAN_WRAPPER
    nullptr,
    instance,
    VK_API_VERSION_1_0,
  };

  VmaAllocator allocator{};
  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator));

  return allocator;
}

VmaAllocationCreateInfo create_allocation_create_info(
    const VmaMemoryUsage usage) {
  return VmaAllocationCreateInfo{
    0u, /* VMA_ALLOCATION_CREATE_MAPPED_BIT - MoltenVK Issue #175 */
        /* VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT */
    usage,
    0u,
    0u,
    0u,
    VK_NULL_HANDLE,
    nullptr,
  };
}

void release_buffer(const Resource::Buffer& buffer) {
  vmaDestroyBuffer(
      buffer.memory.allocator,
      buffer.handle,
      buffer.memory.allocation);
}

void release_image(const Resource::Image& image) {
  if (VK_NULL_HANDLE != image.view) {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(image.memory.allocator, &allocator_info);
    vkDestroyImageView(allocator_info.device, image.view, nullptr);
  }

  vmaDestroyImage(
      image.memory.allocator,
      image.handle,
      image.memory.allocation);
}

} // namespace

void* map(const Resource::Memory& memory) {
  // Call will be ignored by implementation if the memory type this allocation
  // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
  // we want.
  VK_CHECK(vmaInvalidateAllocation(
      memory.allocator, memory.allocation, 0u, VK_WHOLE_SIZE));

  void* data = nullptr;
  VK_CHECK(vmaMapMemory(memory.allocator, memory.allocation, &data));

  return data;
}

Resource::Memory::Scope::Scope(
    const VmaAllocator allocator,
    const VmaAllocation allocation,
    const Access access)
  : allocator_(allocator),
    allocation_(allocation),
    access_(access) {
}

void Resource::Memory::Scope::operator()(const void* const data) const {
  if (C10_UNLIKELY(!data)) {
    return;
  }

  vmaUnmapMemory(allocator_, allocation_);

  if (Access::Write == access_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

Resource::Pool::Pool(
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice device)
  : device_(device),
    allocator_(
        create_allocator(
          instance,
          physical_device,
          device),
        vmaDestroyAllocator) {
    buffers_.reserve(Configuration::kReserve);
    images_.reserve(Configuration::kReserve);
}

Resource::Buffer Resource::Pool::allocate(const Buffer::Descriptor& descriptor) {
  const VkBufferCreateInfo buffer_create_info{
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    nullptr,
    0u,
    descriptor.size,
    descriptor.usage.buffer,
    VK_SHARING_MODE_EXCLUSIVE,
    0u,
    nullptr,
  };

  const VmaAllocationCreateInfo allocation_create_info =
      create_allocation_create_info(descriptor.usage.memory);

  VkBuffer buffer{};
  VmaAllocation allocation{};
  VmaAllocationInfo allocation_info{};

  VK_CHECK(vmaCreateBuffer(
      allocator_.get(),
      &buffer_create_info,
      &allocation_create_info,
      &buffer,
      &allocation,
      &allocation_info));

  buffers_.emplace_back(
      Buffer{
        buffer,
        Memory{
          allocator_.get(),
          allocation,
          allocation_info,
        },
      },
      &release_buffer);

  return buffers_.back().get();
}

Resource::Image Resource::Pool::allocate(const Image::Descriptor& descriptor) {
  const VkImageCreateInfo image_create_info{
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    nullptr,
    0u,
    descriptor.type,
    descriptor.format,
    descriptor.extent,
    1u,
    1u,
    VK_SAMPLE_COUNT_1_BIT,
    VK_IMAGE_TILING_OPTIMAL,
    descriptor.usage.image,
    VK_SHARING_MODE_EXCLUSIVE,
    0u,
    nullptr,
    VK_IMAGE_LAYOUT_UNDEFINED,
  };

  const VmaAllocationCreateInfo allocation_create_info =
      create_allocation_create_info(descriptor.usage.memory);

  VkImage image{};
  VmaAllocation allocation{};
  VmaAllocationInfo allocation_info{};

  VK_CHECK(vmaCreateImage(
      allocator_.get(),
      &image_create_info,
      &allocation_create_info,
      &image,
      &allocation,
      &allocation_info));

  const VkImageViewCreateInfo image_view_create_info{
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
    nullptr,
    0u,
    image,
    descriptor.view.type,
    descriptor.view.format,
    {
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
      VK_COMPONENT_SWIZZLE_IDENTITY,
    },
    {
      VK_IMAGE_ASPECT_COLOR_BIT,
      0u,
      1u,
      0u,
      1u,
    },
  };

  VkImageView view{};
  VK_CHECK(vkCreateImageView(
      device_, &image_view_create_info, nullptr, &view))

  images_.emplace_back(
      Image{
        image,
        view,
        Memory{
          allocator_.get(),
          allocation,
          allocation_info,
        },
      },
      &release_image);

  return images_.back().get();
}

void Resource::Pool::purge() {
  images_.clear();
  buffers_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
