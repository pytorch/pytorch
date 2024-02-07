#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>

#include <mutex>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// CommandBuffer
//

CommandBuffer::CommandBuffer(
    VkCommandBuffer handle,
    const VkCommandBufferUsageFlags flags)
    : handle_(handle),
      flags_(flags),
      state_(CommandBuffer::State::NEW),
      bound_{} {}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : handle_(other.handle_),
      flags_(other.flags_),
      state_(CommandBuffer::State::INVALID),
      bound_(other.bound_) {
  other.handle_ = VK_NULL_HANDLE;
  other.bound_.reset();
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
  VK_CHECK_COND(
      state_ == CommandBuffer::State::NEW,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not NEW.");

  const VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      nullptr,
      flags_,
      nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(handle_, &begin_info));
  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::end() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING ||
          state_ == CommandBuffer::State::SUBMITTED,
      "Vulkan CommandBuffer: called end() on a command buffer whose state "
      "is not RECORDING or SUBMITTED.");

  if (state_ == CommandBuffer::State::RECORDING) {
    VK_CHECK(vkEndCommandBuffer(handle_));
  }
  state_ = CommandBuffer::State::READY;
}

void CommandBuffer::bind_pipeline(
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    const utils::uvec3 local_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called bind_pipeline() on a command buffer whose state "
      "is not RECORDING.");

  if (pipeline != bound_.pipeline) {
    vkCmdBindPipeline(handle_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    bound_.pipeline = pipeline;
  }

  bound_.pipeline_layout = pipeline_layout;
  bound_.local_workgroup_size = local_workgroup_size;

  state_ = CommandBuffer::State::PIPELINE_BOUND;
}

void CommandBuffer::bind_descriptors(VkDescriptorSet descriptors) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::PIPELINE_BOUND,
      "Vulkan CommandBuffer: called bind_descriptors() on a command buffer whose state "
      "is not PIPELINE_BOUND.");

  if (descriptors != bound_.descriptors) {
    vkCmdBindDescriptorSets(
        handle_, // commandBuffer
        VK_PIPELINE_BIND_POINT_COMPUTE, // pipelineBindPoint
        bound_.pipeline_layout, // layout
        0u, // firstSet
        1u, // descriptorSetCount
        &descriptors, // pDescriptorSets
        0u, // dynamicOffsetCount
        nullptr); // pDynamicOffsets
  }

  bound_.descriptors = descriptors;

  state_ = CommandBuffer::State::DESCRIPTORS_BOUND;
}

void CommandBuffer::insert_barrier(const PipelineBarrier& pipeline_barrier) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::DESCRIPTORS_BOUND ||
          state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called insert_barrier() on a command buffer whose state "
      "is not DESCRIPTORS_BOUND or RECORDING.");

  if (pipeline_barrier) {
    std::vector<VkBufferMemoryBarrier> buffer_memory_barriers(4);
    for (const api::BufferMemoryBarrier& memory_barrier :
         pipeline_barrier.buffers) {
      buffer_memory_barriers.push_back(memory_barrier.handle);
    }

    std::vector<VkImageMemoryBarrier> image_memory_barriers(4);
    for (const api::ImageMemoryBarrier& memory_barrier :
         pipeline_barrier.images) {
      image_memory_barriers.push_back(memory_barrier.handle);
    }

    vkCmdPipelineBarrier(
        handle_, // commandBuffer
        pipeline_barrier.stage.src, // srcStageMask
        pipeline_barrier.stage.dst, // dstStageMask
        0u, // dependencyFlags
        0u, // memoryBarrierCount
        nullptr, // pMemoryBarriers
        buffer_memory_barriers.size(), // bufferMemoryBarrierCount
        buffer_memory_barriers.data(), // pMemoryBarriers
        image_memory_barriers.size(), // imageMemoryBarrierCount
        image_memory_barriers.data()); // pImageMemoryBarriers
  }

  state_ = CommandBuffer::State::BARRIERS_INSERTED;
}

void CommandBuffer::dispatch(const utils::uvec3& global_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called dispatch() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  vkCmdDispatch(
      handle_,
      utils::div_up(
          global_workgroup_size.data[0u], bound_.local_workgroup_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u], bound_.local_workgroup_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          bound_.local_workgroup_size.data[2u]));

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_buffer(
    const api::VulkanBuffer& source,
    const api::VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkBufferCopy copy_details{
      src_offset.data[0u], // srcOffset
      dst_offset.data[0u], // dstOffset
      copy_range.data[0u], // size
  };

  vkCmdCopyBuffer(
      handle_, source.handle(), destination.handle(), 1u, &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_texture(
    const api::VulkanImage& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageCopy copy_details{
      src_subresource_layers, // srcSubresource
      create_offset3d(src_offset), // srcOffset
      dst_subresource_layers, // dstSubresource
      create_offset3d(dst_offset), // dstOffset
      create_extent3d(copy_range), // extent
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

void CommandBuffer::copy_texture_to_buffer(
    const api::VulkanImage& source,
    const api::VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkBufferImageCopy copy_details{
      dst_offset.data[0u], // bufferOffset
      dst_offset.data[1u], // bufferRowLength
      dst_offset.data[2u], // bufferImageHeight
      src_subresource_layers, // imageSubresource
      create_offset3d(src_offset), // imageOffset
      create_extent3d(copy_range), // imageExtent
  };

  vkCmdCopyImageToBuffer(
      handle_,
      source.handle(),
      source.layout(),
      destination.handle(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_texture(
    const api::VulkanBuffer& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkBufferImageCopy copy_details{
      src_offset.data[0u], // bufferOffset
      src_offset.data[1u], // bufferRowLength
      src_offset.data[2u], // bufferImageHeight
      dst_subresource_layers, // imageSubresource
      create_offset3d(dst_offset), // imageOffset
      create_extent3d(copy_range), // imageExtent
  };

  vkCmdCopyBufferToImage(
      handle_,
      source.handle(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::write_timestamp(VkQueryPool querypool, const uint32_t idx)
    const {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called write_timestamp() on a command buffer whose state "
      "is not RECORDING.");

  vkCmdWriteTimestamp(
      handle_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, querypool, idx);
}

void CommandBuffer::reset_querypool(
    VkQueryPool querypool,
    const uint32_t first_idx,
    const uint32_t count) const {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called reset_querypool() on a command buffer whose state "
      "is not RECORDING.");

  vkCmdResetQueryPool(handle_, querypool, first_idx, count);
}

VkCommandBuffer CommandBuffer::get_submit_handle(const bool final_use) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::READY,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not READY.");

  VkCommandBuffer handle = handle_;

  if (!is_reusable() || final_use) {
    invalidate();
  }
  state_ = CommandBuffer::State::SUBMITTED;

  return handle;
}

//
// CommandPool
//

CommandPool::CommandPool(
    VkDevice device,
    const uint32_t queue_family_idx,
    const CommandPoolConfig& config)
    : device_(device),
      queue_family_idx_(queue_family_idx),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      buffers_{},
      in_use_(0u) {
  const VkCommandPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      nullptr,
      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      queue_family_idx_,
  };

  VK_CHECK(vkCreateCommandPool(device_, &create_info, nullptr, &pool_));

  // Pre-allocate some command buffers
  allocate_new_batch(config_.cmdPoolInitialSize);
}

CommandPool::~CommandPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  vkDestroyCommandPool(device_, pool_, nullptr);
}

CommandBuffer CommandPool::get_new_cmd(bool reusable) {
  std::lock_guard<std::mutex> lock(mutex_);

  // No-ops if there are command buffers available
  allocate_new_batch(config_.cmdPoolBatchSize);

  VkCommandBuffer handle = buffers_[in_use_];

  VkCommandBufferUsageFlags cmd_flags = 0u;
  if (!reusable) {
    cmd_flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  }

  in_use_++;
  return CommandBuffer(handle, cmd_flags);
}

void CommandPool::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
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
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // commandPool
      VK_COMMAND_BUFFER_LEVEL_PRIMARY, // level
      count, // commandBufferCount
  };

  VK_CHECK(vkAllocateCommandBuffers(
      device_, &allocate_info, buffers_.data() + in_use_));
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
