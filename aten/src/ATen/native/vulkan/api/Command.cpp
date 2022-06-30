#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <mutex>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// CommandBuffer
//

CommandBuffer::CommandBuffer(
    const VkCommandBuffer handle,
    const VkCommandBufferUsageFlags flags)
  : handle_(handle),
    flags_(flags),
    state_(CommandBuffer::State::NEW),
    bound_{} {
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
  : handle_(other.handle_),
    flags_(other.flags_),
    state_(other.state_),
    bound_(other.bound_) {
  other.handle_ = VK_NULL_HANDLE;
  other.bound_.reset();
  state_ = CommandBuffer::State::INVALID;
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
  handle_ = other.handle_;
  flags_ = other.flags_;
  state_ = other.state_;
  bound_ = other.bound_;

  other.handle_ = VK_NULL_HANDLE;
  other.bound_.reset();
  other.state_ = CommandBuffer::State::INVALID;

  return *this;
}

void CommandBuffer::begin() {
  TORCH_CHECK(
      state_ == CommandBuffer::State::NEW,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not NEW.");

  const VkCommandBufferBeginInfo begin_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    nullptr,
    flags_,
    nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(
      handle_,
      &begin_info));
  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::end() {
  TORCH_CHECK(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called end() on a command buffer whose state "
      "is not RECORDING.");

  VK_CHECK(vkEndCommandBuffer(handle_));
  state_ = CommandBuffer::State::READY;
}

void CommandBuffer::bind_pipeline(
    const VkPipeline pipeline,
    const VkPipelineLayout pipeline_layout,
    const utils::uvec3 local_workgroup_size) {
  TORCH_CHECK(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called bind_pipeline() on a command buffer whose state "
      "is not RECORDING.");

  if (pipeline != bound_.pipeline) {
    vkCmdBindPipeline(
        handle_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline);

    bound_.pipeline = pipeline;
  }

  bound_.pipeline_layout = pipeline_layout;
  bound_.local_workgroup_size = local_workgroup_size;

  state_ = CommandBuffer::State::PIPELINE_BOUND;
}

void CommandBuffer::bind_descriptors(const VkDescriptorSet descriptors) {
  TORCH_CHECK(
      state_ == CommandBuffer::State::PIPELINE_BOUND,
      "Vulkan CommandBuffer: called bind_descriptors() on a command buffer whose state "
      "is not PIPELINE_BOUND.");

  if (descriptors != bound_.descriptors) {
    vkCmdBindDescriptorSets(
        handle_,  // commandBuffer
        VK_PIPELINE_BIND_POINT_COMPUTE,  // pipelineBindPoint
        bound_.pipeline_layout,  // layout
        0u,  // firstSet
        1u,  // descriptorSetCount
        &descriptors,  // pDescriptorSets
        0u,  // dynamicOffsetCount
        nullptr);  // pDynamicOffsets
  }

  bound_.descriptors = descriptors;

  state_ = CommandBuffer::State::DESCRIPTORS_BOUND;
}

void CommandBuffer::insert_barrier(const PipelineBarrier& pipeline_barrier) {
  TORCH_CHECK(
      state_ == CommandBuffer::State::DESCRIPTORS_BOUND ||
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called insert_barrier() on a command buffer whose state "
      "is not DESCRIPTORS_BOUND or RECORDING.");

  if (pipeline_barrier) {
    c10::SmallVector<VkBufferMemoryBarrier, 4u> buffer_memory_barriers;
    for (const api::BufferMemoryBarrier& memory_barrier
         : pipeline_barrier.buffers) {
      buffer_memory_barriers.push_back(memory_barrier.handle);
    }

    c10::SmallVector<VkImageMemoryBarrier, 4u> image_memory_barriers;
    for (const api::ImageMemoryBarrier& memory_barrier
         : pipeline_barrier.images) {
      image_memory_barriers.push_back(memory_barrier.handle);
    }

    vkCmdPipelineBarrier(
        handle_,  // commandBuffer
        pipeline_barrier.stage.src,  // srcStageMask
        pipeline_barrier.stage.dst,  // dstStageMask
        0u,  // dependencyFlags
        0u,  // memoryBarrierCount
        nullptr,  // pMemoryBarriers
        buffer_memory_barriers.size(),  // bufferMemoryBarrierCount
        buffer_memory_barriers.data(),  // pMemoryBarriers
        image_memory_barriers.size(),  // imageMemoryBarrierCount
        image_memory_barriers.data());  // pImageMemoryBarriers
  }

  state_ = CommandBuffer::State::BARRIERS_INSERTED;
}

void CommandBuffer::dispatch(const utils::uvec3& global_workgroup_size) {
  TORCH_CHECK(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called dispatch() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  vkCmdDispatch(
      handle_,
      utils::div_up(
          global_workgroup_size.data[0u],
          bound_.local_workgroup_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u],
          bound_.local_workgroup_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          bound_.local_workgroup_size.data[2u]));

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_texture(
    const api::VulkanImage& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  TORCH_CHECK(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
    VK_IMAGE_ASPECT_COLOR_BIT,  // aspectMask
    0u,  // mipLevel
    0u,  // baseArrayLayer
    1u,  // layerCount
  };

  const VkImageSubresourceLayers dst_subresource_layers{
    VK_IMAGE_ASPECT_COLOR_BIT,  // aspectMask
    0u,  // mipLevel
    0u,  // baseArrayLayer
    1u,  // layerCount
  };

  const VkImageCopy copy_details{
    src_subresource_layers,  // srcSubresource
    create_offset3d(src_offset),  // srcOffset
    dst_subresource_layers,  // dstSubresource
    create_offset3d(dst_offset),  // dstOffset
    create_extent3d(copy_range),  // extent
  };

  vkCmdCopyImage(
      handle_,
      source.handle(),
      source.layout(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

VkCommandBuffer CommandBuffer::get_submit_handle() {
  TORCH_CHECK(
      state_ == CommandBuffer::State::READY,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not READY.");

  const VkCommandBuffer handle = handle_;

  handle_ = VK_NULL_HANDLE;
  bound_.reset();
  state_ = CommandBuffer::State::SUBMITTED;

  return handle;
}

//
// CommandPool
//

CommandPool::CommandPool(
    const VkDevice device,
    const uint32_t queue_family_idx,
    const CommandPoolConfig& config)
  : device_(device),
    queue_family_idx_(queue_family_idx),
    pool_(VK_NULL_HANDLE),
    config_(config),
    buffers_{},
    in_use_(0u) {
  const VkCommandPoolCreateInfo create_info{
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    nullptr,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    queue_family_idx_,
  };

  VK_CHECK(vkCreateCommandPool(
      device_,
      &create_info,
      nullptr,
      &pool_));

  // Pre-allocate some command buffers
  allocate_new_batch(config_.cmdPoolInitialSize);
}

CommandPool::CommandPool(CommandPool&& other) noexcept
  : device_(other.device_),
    queue_family_idx_(other.queue_family_idx_),
    pool_(other.pool_),
    config_(other.config_),
    buffers_(std::move(other.buffers_)),
    in_use_(other.in_use_) {
  other.pool_ = VK_NULL_HANDLE;
}

CommandPool::~CommandPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  vkDestroyCommandPool(device_, pool_, nullptr);
}

CommandBuffer CommandPool::get_new_cmd() {
  // No-ops if there are command buffers available
  allocate_new_batch(config_.cmdPoolBatchSize);

  const VkCommandBuffer handle = buffers_[in_use_];

  in_use_++;
  return CommandBuffer(handle);
}

void CommandPool::flush() {
  VK_CHECK(vkResetCommandPool(device_, pool_, 0u));
  in_use_ = 0u;
}

void CommandPool::allocate_new_batch(const uint32_t count) {
  // No-ops if there are still command buffers availble
  if (in_use_ < buffers_.size()) {
    return;
  }

  buffers_.resize(buffers_.size() + count);

  const VkCommandBufferAllocateInfo allocate_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // sType
    nullptr, // pNext
    pool_,  // commandPool
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,  // level
    count,  // commandBufferCount
  };

  VK_CHECK(vkAllocateCommandBuffers(
      device_,
      &allocate_info,
      buffers_.data() + in_use_));
}

namespace {

std::mutex queue_mutex;

VkCommandPool create_command_pool(
    const VkDevice device,
    const uint32_t queue_family_index) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  const VkCommandPoolCreateInfo command_pool_create_info{
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    nullptr,
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    queue_family_index,
  };

  VkCommandPool command_pool{};
  VK_CHECK(vkCreateCommandPool(
      device,
      &command_pool_create_info,
      nullptr,
      &command_pool));

  TORCH_CHECK(
      command_pool,
      "Invalid Vulkan command pool!");

  return command_pool;
}

void allocate_command_buffers(
    const VkDevice device,
    const VkCommandPool command_pool,
    VkCommandBuffer* const command_buffers,
    const uint32_t count) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_pool,
      "Invalid Vulkan command pool!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffers && (count > 0u),
      "Invalid usage!");

  const VkCommandBufferAllocateInfo command_buffer_allocate_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    nullptr,
    command_pool,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    count,
  };

  VK_CHECK(vkAllocateCommandBuffers(
      device,
      &command_buffer_allocate_info,
      command_buffers));
}

} // namespace

Command::Buffer::Buffer(const VkCommandBuffer command_buffer)
  : command_buffer_(command_buffer) {
}

Command::Buffer::Buffer(Buffer&& buffer)
  : command_buffer_(std::move(buffer.command_buffer_)),
    bound_(std::move(buffer.bound_)),
    barriers_(std::move(buffer.barriers_)) {
  buffer.invalidate();
}

Command::Buffer& Command::Buffer::operator=(Buffer&& buffer) {
  if (&buffer != this) {
    command_buffer_ = std::move(buffer.command_buffer_);
    bound_ = std::move(buffer.bound_);
    barriers_ = std::move(buffer.barriers_);

    buffer.invalidate();
  };

  return *this;
}

void Command::Buffer::Buffer::begin() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  const VkCommandBufferBeginInfo command_buffer_begin_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    nullptr,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(
      command_buffer_,
      &command_buffer_begin_info));

  // Reset
  bound_.reset();
  barriers_.reset();
}

void Command::Buffer::Buffer::end() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  VK_CHECK(vkEndCommandBuffer(command_buffer_));
}

void Command::Buffer::barrier(const PipelineBarrier& barrier) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  barriers_.stage.src |= barrier.stage.src;
  barriers_.stage.dst |= barrier.stage.dst;

  barriers_.buffer_barriers.insert(
      barriers_.buffer_barriers.end(),
      barrier.buffers.begin(),
      barrier.buffers.end());

  barriers_.image_barriers.insert(
      barriers_.image_barriers.end(),
      barrier.images.begin(),
      barrier.images.end());
}

void Command::Buffer::bind(
    const VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    utils::uvec3 local_work_group) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  if (pipeline != bound_.pipeline) {
    vkCmdBindPipeline(
        command_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline);

    bound_.pipeline = pipeline;
    bound_.pipeline_layout = pipeline_layout;
    bound_.local_work_group = local_work_group;
  }
}

void Command::Buffer::bind(const Descriptor::Set& set) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  const VkDescriptorSet descriptor_set = set.handle();

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set,
      "Invalid Vulkan descriptor set!");

  if (descriptor_set != bound_.descriptor_set) {
    vkCmdBindDescriptorSets(
        command_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        bound_.pipeline_layout,
        0u,
        1u,
        &descriptor_set,
        0u,
        nullptr);

    bound_.descriptor_set = descriptor_set;
  }
}

void Command::Buffer::copy(
  const api::VulkanBuffer::Package source,
  const api::VulkanBuffer::Package destination) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  barrier();

  const VkBufferCopy buffer_copy{
    0u,
    0u,
    std::min(source.buffer_range, destination.buffer_range),
  };

  vkCmdCopyBuffer(
      command_buffer_,
      source.handle,
      destination.handle,
      1u,
      &buffer_copy);
}

void Command::Buffer::copy_image(
  const api::VulkanImage& source,
  const api::VulkanImage& destination,
  const api::utils::uvec3& src_offset,
  const api::utils::uvec3& dst_offset,
  const api::utils::uvec3& copy_range) {

  barrier();

  const VkImageSubresourceLayers src_subresource_layers{
    VK_IMAGE_ASPECT_COLOR_BIT,  // aspectMask
    0u,  // mipLevel
    0u,  // baseArrayLayer
    1u,  // layerCount
  };

  const VkImageSubresourceLayers dst_subresource_layers{
    VK_IMAGE_ASPECT_COLOR_BIT,  // aspectMask
    0u,  // mipLevel
    0u,  // baseArrayLayer
    1u,  // layerCount
  };

  const VkImageCopy copy_details{
    src_subresource_layers,  // srcSubresource
    create_offset3d(src_offset),  // srcOffset
    dst_subresource_layers,  // dstSubresource
    create_offset3d(dst_offset),  // dstOffset
    create_extent3d(copy_range),  // extent
  };

  vkCmdCopyImage(
      command_buffer_,
      source.handle(),
      source.layout(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);
}

void Command::Buffer::dispatch(
    const utils::uvec3& global_work_group) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  barrier();

  vkCmdDispatch(
      command_buffer_,
      utils::div_up(
          global_work_group.data[0u],
          bound_.local_work_group.data[0u]),
      utils::div_up(
          global_work_group.data[1u],
          bound_.local_work_group.data[1u]),
      utils::div_up(
          global_work_group.data[2u],
          bound_.local_work_group.data[2u]));
}

void Command::Buffer::barrier() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  if (barriers_.stage) {
    c10::SmallVector<VkBufferMemoryBarrier, 4u> buffer_memory_barriers;

    for (const api::BufferMemoryBarrier& memory_barrier
         : barriers_.buffer_barriers) {
      buffer_memory_barriers.push_back(memory_barrier.handle);
    }

    c10::SmallVector<VkImageMemoryBarrier, 4u> image_memory_barriers;

    for (const api::ImageMemoryBarrier& memory_barrier
         : barriers_.image_barriers) {
      image_memory_barriers.push_back(memory_barrier.handle);
    }

    vkCmdPipelineBarrier(
        command_buffer_,
        barriers_.stage.src,
        barriers_.stage.dst,
        0u,
        0u,
        nullptr,
        buffer_memory_barriers.size(),
        buffer_memory_barriers.data(),
        image_memory_barriers.size(),
        image_memory_barriers.data());
  }

  // Reset
  barriers_.reset();
}

void Command::Buffer::invalidate() {
  command_buffer_ = VK_NULL_HANDLE;
}

inline void Command::Buffer::Bound::reset() {
  pipeline = {};
  pipeline_layout = {};
  local_work_group = {};
  descriptor_set = VK_NULL_HANDLE;
}

inline Command::Buffer::Barrier::Stages::operator bool() const {
  return (0u != src) || (0u != dst);
}

inline void Command::Buffer::Barrier::reset() {
  stage = {};
  buffer_barriers.clear();
  image_barriers.clear();
}

Command::Pool::Pool(const GPU& gpu)
  : device_(gpu.device),
    command_pool_(
        create_command_pool(gpu.device, gpu.queue_family_index),
        VK_DELETER(CommandPool)(device_)),
    buffer_{} {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_pool_,
      "Invalid Vulkan command pool!");

  buffer_.pool.reserve(Configuration::kReserve);
}

Command::Pool::Pool(Pool&& pool)
  : device_(std::move(pool.device_)),
    command_pool_(std::move(pool.command_pool_)),
    buffer_(std::move(pool.buffer_)),
    stream_(std::move(pool.stream_)) {
  pool.invalidate();
}

Command::Pool& Command::Pool::operator=(Pool&& pool) {
  if (&pool != this) {
    device_ = std::move(pool.device_);
    command_pool_ = std::move(pool.command_pool_);
    buffer_ = std::move(pool.buffer_);
    stream_ = std::move(pool.stream_);

    pool.invalidate();
  };

  return *this;
}

Command::Pool::~Pool() {
  try {
    if (device_ && command_pool_) {
      purge();
    }
  }
  catch (const std::exception& e) {
    TORCH_WARN(
        "Vulkan: Command pool destructor raised an exception! Error: ",
        e.what());
  }
  catch (...) {
    TORCH_WARN(
        "Vulkan: Command pool destructor raised an exception! "
        "Error: Unknown");
  }
}

Command::Buffer Command::Pool::allocate() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && command_pool_,
      "This command pool is in an invalid state! "
      "Potential reason: This command pool is moved from.");

  if (buffer_.pool.size() == buffer_.in_use) {
    buffer_.pool.resize(
        buffer_.pool.size() +
        Configuration::kQuantum);

    allocate_command_buffers(
        device_,
        command_pool_.get(),
        buffer_.pool.data() + buffer_.in_use,
        Configuration::kQuantum);
  }

  return Buffer(buffer_.pool[buffer_.in_use++]);
}

Command::Buffer& Command::Pool::stream() {
  if (!stream_.buffer) {
    stream_.buffer = allocate();
    stream_.buffer.begin();
    stream_.counter = 0u;
  }

  return stream_.buffer;
}

void Command::Pool::purge() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && command_pool_,
      "This command pool is in an invalid state! "
      "Potential reason: This command pool is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !stream_.buffer,
      "Pending command buffer detected.  Make sure all command buffers are "
      "submitted to the queue for execution prior to reclaiming pool memory.");

  buffer_.in_use = 0u;
  VK_CHECK(vkResetCommandPool(device_, command_pool_.get(), 0u));
}

void Command::Pool::submit(
    const VkQueue queue,
    const c10::ArrayRef<const Buffer> buffers,
    const VkFence fence) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_ && command_pool_,
      "This command pool is in an invalid state! "
      "Potential reason: This command pool is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      queue,
      "Invalid Vulkan queue!");

  c10::SmallVector<VkCommandBuffer, Configuration::kReserve> command_buffers;
  command_buffers.reserve(buffers.size());

  for (const Buffer& buffer : buffers) {
    VkCommandBuffer command_buffer = buffer.handle();

    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        command_buffer,
        "Invalid Vulkan command buffer!");

    // Are we submitting our one and only command stream, or a regular command
    // buffer whose scope is manually maintained by the user?  Automatically
    // maintain state and submission rate if the former.

    if (stream_.buffer.handle() == command_buffer) {
      // Hand the stream off to the driver if:
      // - The user has implictly signaled interest in the results via a fence.
      // - We are over the submission cutoff.  We don't want to starve the GPU.

      if (fence != VK_NULL_HANDLE || (stream_.counter++ > Configuration::kSubmit)) {
        stream_.buffer.end();
        stream_.buffer.invalidate();
      }
      // Skip - Accumulate more calls prior to submission.
      else {
        command_buffer = VK_NULL_HANDLE;
      }
    }

    if (command_buffer) {
      command_buffers.push_back(command_buffer);
    }
  }

  if (!command_buffers.empty()) {
    const VkSubmitInfo submit_info{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,
      nullptr,
      0u,
      nullptr,
      nullptr,
      utils::safe_downcast<uint32_t>(command_buffers.size()),
      command_buffers.data(),
      0u,
      nullptr,
    };

    {
      // vkQueueSubmit is not thread-safe, only one thread can push the commands at a time.
      // (See https://vkguide.dev/docs/chapter-1/vulkan_command_flow/#vulkan-command-execution)
      // The number of available queues depends on GPU. It could be 1 and we cannot assume we can create multiple queues.
      // Thus, we need to avoid calling vkQueueSubmit from multiple threads at the same time.
      // When running Vulkan backend in different threads without any locking mechanism,
      // vkQueueSubmit will get the VK_ERROR_INITIALIZATION_FAILED(-3) error.
      std::lock_guard<std::mutex> guard(queue_mutex);
      VK_CHECK(vkQueueSubmit(queue, 1u, &submit_info, fence));
    }
  }
}

void Command::Pool::invalidate() {
  device_ = VK_NULL_HANDLE;
  command_pool_.reset();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
