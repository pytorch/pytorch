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
    nullptr,
    nullptr,
    instance,
    VK_API_VERSION_1_0,
  };

  VmaAllocator allocator{};
  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator));
  TORCH_CHECK(allocator, "Invalid VMA (Vulkan Memory Allocator) allocator!");

  return allocator;
}

VmaAllocationCreateInfo create_allocation_create_info(
    const Resource::Memory::Descriptor& descriptor) {
  return VmaAllocationCreateInfo{
    VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT |
        /* VMA_ALLOCATION_CREATE_MAPPED_BIT - MoltenVK Issue #175 */
        0,
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

void* map(
    const Resource::Memory& memory,
    const Resource::Memory::Access::Flags access) {
  void* data = nullptr;
  VK_CHECK(vmaMapMemory(memory.allocator, memory.allocation, &data));

  if (access & Resource::Memory::Access::Read) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(vmaInvalidateAllocation(
        memory.allocator, memory.allocation, 0u, VK_WHOLE_SIZE));
  }

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
      "Invalid VMA (Vulkan Memory Allocator) allocator!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocation,
      "Invalid VMA (Vulkan Memory Allocator) allocation!");
}

void Resource::Memory::Scope::operator()(const void* const data) const {
  if (C10_UNLIKELY(!data)) {
    return;
  }

  if (access_ & Access::Write) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }

  vmaUnmapMemory(allocator_, allocation_);
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
    1.0f,
    VK_FALSE,
    VK_COMPARE_OP_NEVER,
    0.0f,
    VK_LOD_CLAMP_NONE,
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

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      fence,
      "Invalid Vulkan fence!");

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

namespace {

class Linear final : public Resource::Pool::Policy {
 public:
  Linear(
      VkDeviceSize block_size,
      uint32_t min_block_count,
      uint32_t max_block_count);

  virtual void enact(
      VmaAllocator allocator,
      const VkMemoryRequirements& memory_requirements,
      VmaAllocationCreateInfo& allocation_create_info) override;

 private:
  struct Configuration final {
    static constexpr uint32_t kReserve = 16u;
  };

  struct Entry final {
    class Deleter final {
     public:
      explicit Deleter(VmaAllocator);
      void operator()(VmaPool) const;

     private:
      VmaAllocator allocator_;
    };

    uint32_t memory_type_index;
    Handle<VmaPool, Deleter> handle;
  };

  std::vector<Entry> pools_;

  struct {
    VkDeviceSize size;
    uint32_t min;
    uint32_t max;
  } block_;
};

Linear::Entry::Deleter::Deleter(const VmaAllocator allocator)
  : allocator_(allocator) {
}

void Linear::Entry::Deleter::operator()(const VmaPool pool) const {
  vmaDestroyPool(allocator_, pool);
}

Linear::Linear(
    const VkDeviceSize block_size,
    const uint32_t min_block_count,
    const uint32_t max_block_count)
  : block_ {
      block_size,
      min_block_count,
      max_block_count,
    } {
  pools_.reserve(Configuration::kReserve);
}

void Linear::enact(
    const VmaAllocator allocator,
    const VkMemoryRequirements& memory_requirements,
    VmaAllocationCreateInfo& allocation_create_info) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      allocator,
      "Invalid VMA (Vulkan Memory Allocator) allocator!");

  uint32_t memory_type_index = 0u;
  VK_CHECK(vmaFindMemoryTypeIndex(
      allocator,
      memory_requirements.memoryTypeBits,
      &allocation_create_info,
      &memory_type_index));

  auto pool_itr = std::find_if(
      pools_.begin(),
      pools_.end(),
      [memory_type_index](const Entry& entry) {
    return entry.memory_type_index == memory_type_index;
  });

  if (pools_.end() == pool_itr) {
    const VmaPoolCreateInfo pool_create_info{
      memory_type_index,
      VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT,
      block_.size,
      block_.min,
      block_.max,
      0u,
    };

    VmaPool pool{};
    VK_CHECK(vmaCreatePool(
        allocator,
        &pool_create_info,
        &pool));

    TORCH_CHECK(
        pool,
        "Invalid VMA (Vulkan Memory Allocator) memory pool!");

    pools_.push_back({
      memory_type_index,
      {
        pool,
        Entry::Deleter(allocator),
      },
    });

    pool_itr = std::prev(pools_.end());
  }

  allocation_create_info.pool = pool_itr->handle.get();
}

} // namespace

std::unique_ptr<Resource::Pool::Policy> Resource::Pool::Policy::linear(
    const VkDeviceSize block_size,
    const uint32_t min_block_count,
    const uint32_t max_block_count) {
  return std::make_unique<Linear>(
      block_size,
      min_block_count,
      max_block_count);
}

Resource::Pool::Pool(
    const GPU& gpu,
    std::unique_ptr<Policy> policy)
  : device_(gpu.device),
    allocator_(
        create_allocator(
            gpu.adapter->runtime->instance(),
            gpu.adapter->handle,
            device_),
        vmaDestroyAllocator),
    memory_{
      std::move(policy),
    },
    image_{
      .sampler = Image::Sampler{gpu},
    },
    fence_{} {
  buffer_.pool.reserve(Configuration::kReserve);
  image_.pool.reserve(Configuration::kReserve);
  fence_.pool.reserve(Configuration::kReserve);
}

Resource::Pool::Pool(Pool&& pool)
  : device_(std::move(pool.device_)),
    allocator_(std::move(pool.allocator_)),
    memory_(std::move(pool.memory_)),
    buffer_(std::move(pool.buffer_)),
    image_(std::move(pool.image_)),
    fence_(std::move(pool.fence_)) {
  pool.invalidate();
}

Resource::Pool& Resource::Pool::operator=(Pool&& pool) {
  if (&pool != this) {
    device_ = std::move(pool.device_);
    allocator_ = std::move(pool.allocator_);
    memory_ = std::move(pool.memory_);
    buffer_ = std::move(pool.buffer_);
    image_ = std::move(pool.image_);
    fence_ = std::move(pool.fence_);

    pool.invalidate();
  };

  return *this;
}

Resource::Pool::~Pool() {
  try {
    if (device_ && allocator_) {
      purge();
    }
  }
  catch (const std::exception& e) {
    TORCH_WARN(
        "Vulkan: Resource pool destructor raised an exception! Error: ",
        e.what());
  }
  catch (...) {
    TORCH_WARN(
        "Vulkan: Resource pool destructor raised an exception! "
        "Error: Unknown");
  }
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

  VkBuffer buffer{};
  VK_CHECK(vkCreateBuffer(
      device_,
      &buffer_create_info,
      nullptr,
      &buffer));

  TORCH_CHECK(
      buffer,
      "Invalid Vulkan buffer!");

  VkMemoryRequirements memory_requirements{};
  vkGetBufferMemoryRequirements(
      device_,
      buffer,
      &memory_requirements);

  VmaAllocationCreateInfo allocation_create_info =
      create_allocation_create_info(descriptor.usage.memory);

  if (memory_.policy) {
    memory_.policy->enact(
        allocator_.get(),
        memory_requirements,
        allocation_create_info);
  }

  VmaAllocation allocation{};
  VK_CHECK(vmaAllocateMemory(
      allocator_.get(),
      &memory_requirements,
      &allocation_create_info,
      &allocation,
      nullptr));

  TORCH_CHECK(
      allocation,
      "Invalid VMA (Vulkan Memory Allocator) allocation!");

  VK_CHECK(vmaBindBufferMemory(
      allocator_.get(),
      allocation,
      buffer));

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

  VkImage image{};
  VK_CHECK(vkCreateImage(
      device_,
      &image_create_info,
      nullptr,
      &image));

  TORCH_CHECK(
      image,
      "Invalid Vulkan image!");

  VkMemoryRequirements memory_requirements{};
  vkGetImageMemoryRequirements(
      device_,
      image,
      &memory_requirements);

  VmaAllocationCreateInfo allocation_create_info =
      create_allocation_create_info(descriptor.usage.memory);

  if (memory_.policy) {
    memory_.policy->enact(
        allocator_.get(),
        memory_requirements,
        allocation_create_info);
  }

  VmaAllocation allocation{};
  VK_CHECK(vmaAllocateMemory(
      allocator_.get(),
      &memory_requirements,
      &allocation_create_info,
      &allocation,
      nullptr));

  TORCH_CHECK(
      allocation,
      "Invalid VMA (Vulkan Memory Allocator) allocation!");

  VK_CHECK(vmaBindImageMemory(
      allocator_.get(),
      allocation,
      image));

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
      VK_REMAINING_MIP_LEVELS,
      0u,
      VK_REMAINING_ARRAY_LAYERS,
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

void Resource::Pool::invalidate() {
  device_ = VK_NULL_HANDLE;
  allocator_.reset();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
