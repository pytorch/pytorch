#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Adapter.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

VkCommandPool create_command_pool(
    const VkDevice device,
    uint32_t queue_family_index) {
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

VkCommandBuffer allocate_command_buffer(
    const VkDevice device,
    const VkCommandPool command_pool) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_pool,
      "Invalid Vulkan command pool!");

  const VkCommandBufferAllocateInfo command_buffer_allocate_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    nullptr,
    command_pool,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    1u,
  };

  VkCommandBuffer command_buffer{};
  VK_CHECK(vkAllocateCommandBuffers(
      device,
      &command_buffer_allocate_info,
      &command_buffer));

  TORCH_CHECK(
      command_buffer,
      "Invalid Vulkan command buffer!");

  return command_buffer;
}

} // namespace

Command::Buffer::Buffer(
    const VkDevice device,
    const VkCommandPool command_pool)
  : command_buffer_(allocate_command_buffer(device, command_pool)) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_buffer_,
      "Invalid Vulkan command buffer!");
}

void Command::Buffer::Buffer::begin() {
  const VkCommandBufferBeginInfo command_buffer_begin_info{
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    nullptr,
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(
      command_buffer_,
      &command_buffer_begin_info));
}

void Command::Buffer::Buffer::end() {
  VK_CHECK(vkEndCommandBuffer(command_buffer_));
}

void Command::Buffer::bind(const VkPipeline pipeline) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      pipeline,
      "Invalid Vulkan pipeline!");

  vkCmdBindPipeline(
      command_buffer_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline);
}

void Command::Buffer::bind(
    const VkPipelineLayout pipeline_layout,
    const VkDescriptorSet descriptor_set) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      pipeline_layout,
      "Invalid Vulkan pipeline layout!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      descriptor_set,
      "Invalid Vulkan descriptor set!");

  vkCmdBindDescriptorSets(
      command_buffer_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0u,
      1u,
      &descriptor_set,
      0u,
      nullptr);
}

void Command::Buffer::copy(
    const VkBuffer source,
    const VkBuffer destination,
    const size_t size) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      source,
      "Invalid Vulkan source buffer!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      destination,
      "Invalid Vulkan destination buffer!");

  const VkBufferCopy buffer_copy{
    0u,
    0u,
    size,
  };

  vkCmdCopyBuffer(
      command_buffer_,
      source,
      destination,
      1u,
      &buffer_copy);
}

void Command::Buffer::dispatch(
    const Shader::WorkGroup& work_group) {
  vkCmdDispatch(
      command_buffer_,
      work_group.x,
      work_group.y,
      work_group.z);
}

void Command::Buffer::submit(
    const VkQueue queue,
    const VkFence fence) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      queue,
      "Invalid Vulkan queue!");

  const VkSubmitInfo submit_info{
    VK_STRUCTURE_TYPE_SUBMIT_INFO,
    nullptr,
    0u,
    nullptr,
    nullptr,
    1u,
    &command_buffer_,
    0u,
    nullptr,
  };

  VK_CHECK(vkQueueSubmit(queue, 1u, &submit_info, fence));
}

Command::Pool::Pool(const GPU& gpu)
  : device_(gpu.device),
    command_pool_(
        create_command_pool(gpu.device, gpu.adapter->compute_queue_family_index),
        VK_DELETER(CommandPool)(device_)) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      command_pool_,
      "Invalid Vulkan command pool!");
}

Command::Buffer Command::Pool::buffer() {
  return Buffer(device_, command_pool_.get());
}

void Command::Pool::purge() {
  VK_CHECK(vkResetCommandPool(device_, command_pool_.get(), 0u));
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
