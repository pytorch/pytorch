#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Adapter.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VmaAllocator create_allocator(
    const VkInstance instance,
    const VkPhysicalDevice physical_device,
    const VkDevice device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      instance,
      "Invalid Vulkan instance!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      physical_device,
      "Invalid Vulkan physical device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

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
  TORCH_CHECK(allocator, "Invalid VMA allocator!");

  return allocator;
}

VmaAllocationCreateInfo create_allocation_create_info(
    const Resource::Memory::Descriptor& descriptor) {
  return VmaAllocationCreateInfo{
    0u, /* VMA_ALLOCATION_CREATE_MAPPED_BIT - MoltenVK Issue #175 */
        /* VMA_ALLOCATION_CREATE_STRATEGY_MIN_FRAGMENTATION_BIT */
    descriptor.usage,
    descriptor.required,
    descriptor.preferred,
    0u,
    VK_NULL_HANDLE,
    nullptr,
  };
}

void release_buffer(const Resource::Buffer& buffer) {
  // Safe to pass null as buffer or allocation.
  vmaDestroyBuffer(
      buffer.memory.allocator,
      buffer.object.handle,
      buffer.memory.allocation);
}

void release_image(const Resource::Image& image) {
  // Sampler is an immutable object. Its lifetime is managed through the cache.

  if (VK_NULL_HANDLE != image.object.view) {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(image.memory.allocator, &allocator_info);
    vkDestroyImageView(allocator_info.device, image.object.view, nullptr);
  }

  // Safe to pass null as image or allocation.
  vmaDestroyImage(
      image.memory.allocator,
      image.object.handle,
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
    const Access::Flags access)
  : allocator_(allocator),
    allocation_(allocation),
    access_(access) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator,
      "Invalid VMA allocator!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocation,
      "Invalid VMA allocation!");
}

void Resource::Memory::Scope::operator()(const void* const data) const {
  if (C10_UNLIKELY(!data)) {
    return;
  }

  vmaUnmapMemory(allocator_, allocation_);

  if (access_ & Access::Write) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

Resource::Image::Sampler::Factory::Factory(const GPU& gpu)
  : device_(gpu.device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");
}

typename Resource::Image::Sampler::Factory::Handle
Resource::Image::Sampler::Factory::operator()(
    const Descriptor& descriptor) const {
  const VkSamplerCreateInfo sampler_create_info{
    VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
    nullptr,
    0u,
    descriptor.filter,
    descriptor.filter,
    descriptor.mipmap_mode,
    descriptor.address_mode,
    descriptor.address_mode,
    descriptor.address_mode,
    0.0f,
    VK_FALSE,
    0.0f,
    VK_FALSE,
    VK_COMPARE_OP_NEVER,
    0.0f,
    0.0f,
    descriptor.border,
    VK_FALSE,
  };

  VkSampler sampler{};
  VK_CHECK(vkCreateSampler(
      device_,
      &sampler_create_info,
      nullptr,
      &sampler));

  TORCH_CHECK(
      sampler,
      "Invalid Vulkan image sampler!");

  return Handle{
    sampler,
    Deleter(device_),
  };
}

VkFence Resource::Fence::handle(const bool add_to_waitlist) const {
  if (!pool) {
    return VK_NULL_HANDLE;
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      id < pool->fence_.pool.size(),
      "Invalid Vulkan fence!");

  const VkFence fence = pool->fence_.pool[id].get();
  if (add_to_waitlist) {
    pool->fence_.waitlist.push_back(fence);
  }

  return fence;
}

void Resource::Fence::wait(const uint64_t timeout_nanoseconds) {
  const VkFence fence = handle(/* add_to_waitlist = */ false);

  const auto waitlist_itr = std::find(
      pool->fence_.waitlist.cbegin(),
      pool->fence_.waitlist.cend(),
      fence);

  if (pool->fence_.waitlist.cend() != waitlist_itr) {
    VK_CHECK(vkWaitForFences(
        pool->device_,
        1u,
        &fence,
        VK_TRUE,
        timeout_nanoseconds));

    VK_CHECK(vkResetFences(
        pool->device_,
        1u,
        &fence));

    pool->fence_.waitlist.erase(waitlist_itr);
  }
}

Resource::Pool::Pool(const GPU& gpu)
  : device_(gpu.device),
    allocator_(
        create_allocator(
          gpu.adapter->runtime->instance(),
          gpu.adapter->handle,
          device_),
        vmaDestroyAllocator),
    buffer_{},
    image_{
      .sampler = Image::Sampler{gpu},
    },
    fence_{} {
  buffer_.pool.reserve(Configuration::kReserve);
  image_.pool.reserve(Configuration::kReserve);
  fence_.pool.reserve(Configuration::kReserve);
}

Resource::Pool::~Pool() {
  try {
    purge();
  }
  catch (...) {
  }
}

Resource::Pool::Pool(Pool&& pool)
  : device_(std::move(pool.device_)),
    allocator_(std::move(pool.allocator_)),
    buffer_(std::move(pool.buffer_)),
    image_(std::move(pool.image_)),
    fence_(std::move(pool.fence_)) {
  pool.device_ = VK_NULL_HANDLE;
}

Resource::Pool& Resource::Pool::operator=(Pool&& pool) {
  if (&pool != this) {
    device_ = std::move(pool.device_);
    allocator_ = std::move(pool.allocator_);
    buffer_ = std::move(pool.buffer_);
    image_ = std::move(pool.image_);
    fence_ = std::move(pool.fence_);

    pool.device_ = VK_NULL_HANDLE;
  };

  return *this;
}

Resource::Buffer Resource::Pool::buffer(
    const Buffer::Descriptor& descriptor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && allocator_,
      "This resource pool is in an invalid state! ",
      "Potential reason: This resource pool is moved from.");

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

  TORCH_CHECK(buffer, "Invalid Vulkan buffer!");
  TORCH_CHECK(allocation, "Invalid VMA allocation!");

  buffer_.pool.emplace_back(
      Buffer{
        Buffer::Object{
          buffer,
          0u,
          descriptor.size,
        },
        Memory{
          allocator_.get(),
          allocation,
        },
      },
      &release_buffer);

  return buffer_.pool.back().get();
}

Resource::Image Resource::Pool::image(
    const Image::Descriptor& descriptor) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && allocator_,
      "This resource pool is in an invalid state! ",
      "Potential reason: This resource pool is moved from.");

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

  TORCH_CHECK(image, "Invalid Vulkan image!");
  TORCH_CHECK(allocation, "Invalid VMA allocation!");

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
      device_,
      &image_view_create_info,
      nullptr,
      &view));

  TORCH_CHECK(
      view,
      "Invalid Vulkan image view!");

  image_.pool.emplace_back(
      Image{
        Image::Object{
          image,
          VK_IMAGE_LAYOUT_UNDEFINED,
          view,
          image_.sampler.cache.retrieve(descriptor.sampler),
        },
        Memory{
          allocator_.get(),
          allocation,
        },
      },
      &release_image);

  return image_.pool.back().get();
}

Resource::Fence Resource::Pool::fence() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && allocator_,
      "This resource pool is in an invalid state! ",
      "Potential reason: This resource pool is moved from.");

  if (fence_.pool.size() == fence_.in_use) {
    const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      nullptr,
      0u,
    };

    VkFence fence{};
    VK_CHECK(vkCreateFence(
        device_,
        &fence_create_info,
        nullptr,
        &fence));

    TORCH_CHECK(
        fence,
        "Invalid Vulkan fence!");

    fence_.pool.emplace_back(fence, VK_DELETER(Fence)(device_));
  }

  return Fence{
    this,
    fence_.in_use++,
  };
}

void Resource::Pool::purge() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && allocator_,
      "This resource pool is in an invalid state! ",
      "Potential reason: This resource pool is moved from.");

  if (!fence_.waitlist.empty()) {
    VK_CHECK(vkWaitForFences(
        device_,
        fence_.waitlist.size(),
        fence_.waitlist.data(),
        VK_TRUE,
        UINT64_MAX));

    VK_CHECK(vkResetFences(
        device_,
        fence_.waitlist.size(),
        fence_.waitlist.data()));

    fence_.waitlist.clear();
  }

  fence_.in_use = 0u;
  image_.pool.clear();
  buffer_.pool.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
