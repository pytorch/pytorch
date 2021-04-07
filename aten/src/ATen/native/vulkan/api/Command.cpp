#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

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

void Command::Buffer::barrier(const Pipeline::Barrier& barrier) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  barriers_.stage.src |= barrier.stage.src;
  barriers_.stage.dst |= barrier.stage.dst;

  barriers_.buffers.insert(
      barriers_.buffers.end(),
      barrier.buffers.begin(),
      barrier.buffers.end());

  barriers_.images.insert(
      barriers_.images.end(),
      barrier.images.begin(),
      barrier.images.end());
}

void Command::Buffer::bind(const Pipeline::Object& pipeline) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      pipeline,
      "Invalid Vulkan pipeline!");

  if (pipeline.handle != bound_.pipeline.handle) {
    vkCmdBindPipeline(
        command_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.handle);

    bound_.pipeline = pipeline;
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
        bound_.pipeline.layout,
        0u,
        1u,
        &descriptor_set,
        0u,
        nullptr);

    bound_.descriptor_set = descriptor_set;
  }
}

void Command::Buffer::copy(
    const Resource::Buffer::Object source,
    const Resource::Buffer::Object destination) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      source,
      "Invalid Vulkan source buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      destination,
      "Invalid Vulkan destination buffer!");

  barrier();

  const VkBufferCopy buffer_copy{
    0u,
    0u,
    std::min(source.range, destination.range),
  };

  vkCmdCopyBuffer(
      command_buffer_,
      source.handle,
      destination.handle,
      1u,
      &buffer_copy);
}

void Command::Buffer::dispatch(
    const Shader::WorkGroup& global_work_group) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  barrier();

  vkCmdDispatch(
      command_buffer_,
      utils::div_up(
          global_work_group.data[0u],
          bound_.pipeline.local_work_group.data[0u]),
      utils::div_up(
          global_work_group.data[1u],
          bound_.pipeline.local_work_group.data[1u]),
      utils::div_up(
          global_work_group.data[2u],
          bound_.pipeline.local_work_group.data[2u]));
}

void Command::Buffer::barrier() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "This command buffer is in an invalid state! "
      "Potential reason: This command buffer is moved from.");

  if (barriers_.stage) {
    c10::SmallVector<VkBufferMemoryBarrier, 4u> buffer_memory_barriers;

    for (const Resource::Buffer::Barrier& barrier : barriers_.buffers) {
      buffer_memory_barriers.push_back({
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            barrier.memory.src,
            barrier.memory.dst,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            barrier.object.handle,
            barrier.object.offset,
            barrier.object.range,
          });
    }

    c10::SmallVector<VkImageMemoryBarrier, 4u> image_memory_barriers;

    for (const Resource::Image::Barrier& barrier : barriers_.images) {
      image_memory_barriers.push_back({
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            barrier.memory.src,
            barrier.memory.dst,
            barrier.layout.src,
            barrier.layout.dst,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            barrier.object.handle,
            {
              VK_IMAGE_ASPECT_COLOR_BIT,
              0u,
              VK_REMAINING_MIP_LEVELS,
              0u,
              VK_REMAINING_ARRAY_LAYERS,
            },
          });
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
  descriptor_set = VK_NULL_HANDLE;
}

inline Command::Buffer::Barrier::Stage::operator bool() const {
  return (0u != src) || (0u != dst);
}

inline void Command::Buffer::Barrier::reset() {
  stage = {};
  buffers.clear();
  images.clear();
}

Command::Pool::Pool(const GPU& gpu)
  : device_(gpu.device),
    command_pool_(
        create_command_pool(gpu.device, gpu.adapter->compute_queue_family_index),
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
    const Resource::Fence fence) {
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

      if (fence || (stream_.counter++ > Configuration::kSubmit)) {
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
      command_buffers.size(),
      command_buffers.data(),
      0u,
      nullptr,
    };

    VK_CHECK(vkQueueSubmit(queue, 1u, &submit_info, fence.handle()));
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
