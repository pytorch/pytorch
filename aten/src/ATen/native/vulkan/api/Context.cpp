#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/vulkan/Context.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Context::Context(
    const VkInstance instance, size_t adapter_i, const ContextConfig config)
    : config_(config),
      instance_(instance),
      adapter_p_(runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      command_pool_(device_, queue_.family_index, config_.cmdPoolConfig),
      command_(gpu()),
      descriptor_(gpu()),
      fences_(device_),
      querypool_(
        device_,
        adapter_p_->timestamp_compute_and_graphics(),
        adapter_p_->timestamp_period()),
      cmd_(VK_NULL_HANDLE),
      submit_count_{0u},
      fence_{},
      cmd_submit_list_{},
      buffers_to_clear_{},
      images_to_clear_{} {
}

Context::~Context() {
  flush();
  // Let the device know the context is done with the queue
  adapter_p_->return_queue(queue_);
}

Descriptor::Set Context::submit_compute_prologue(
    CommandBuffer& command_buffer,
    const ShaderLayout::Signature& shader_layout_signature,
    const ShaderSource& shader_descriptor,
    const utils::uvec3& local_workgroup_size) {

  const VkDescriptorSetLayout shader_layout = \
      shader_layout_cache().retrieve(shader_layout_signature);

  const VkPipelineLayout pipeline_layout = \
      pipeline_layout_cache().retrieve(shader_layout);

  const VkPipeline pipeline = pipeline_cache().retrieve({
      pipeline_layout_cache().retrieve(shader_layout),
      shader_cache().retrieve(shader_descriptor),
      local_workgroup_size});

  command_buffer.bind_pipeline(
      pipeline, pipeline_layout, local_workgroup_size);

  return descriptor().pool.allocate(shader_layout, shader_layout_signature);
}

void Context::submit_compute_epilogue(
    CommandBuffer& command_buffer,
    const Descriptor::Set& descriptors,
    const PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_workgroup_size) {
  command_buffer.bind_descriptors(descriptors.handle());
  command_buffer.insert_barrier(pipeline_barrier);

  command_buffer.dispatch(global_workgroup_size);
}

void Context::submit_texture_copy(
    const PipelineBarrier& pipeline_barrier,
    const api::VulkanImage& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset,
    const VkFence fence_handle) {
  set_cmd();

  cmd_.insert_barrier(pipeline_barrier);

  cmd_.copy_texture_to_texture(
      source, destination, copy_range, src_offset, dst_offset);

  submit_count_++;
  if (fence_handle != VK_NULL_HANDLE ||
      submit_count_ >= config_.cmdSubmitFrequency) {
    submit_cmd_to_gpu(fence_handle);
  }
}

void Context::submit_cmd_to_gpu(const VkFence fence_handle) {
  if (cmd_) {
    cmd_.end();
    adapter_p_->submit_cmd(queue_, cmd_.get_submit_handle(), fence_handle);

    submit_count_ = 0u;
  }
}

void Context::flush() {
  VK_CHECK(vkQueueWaitIdle(queue()));

  command_pool_.flush();

  descriptor().pool.purge();
  command().pool.purge();

  buffers_to_clear_.clear();
  images_to_clear_.clear();
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const ContextConfig config{
        16u,  // cmdSubmitFrequency
        {  // cmdPoolConfig
          32u,  // cmdPoolInitialSize
          8u,  // cmdPoolBatchSize
        },
      };
      return new Context(
          runtime()->instance(), runtime()->default_adapter_i(), config);
    }
    catch (const std::exception& e) {
      TORCH_CHECK(false, "Vulkan: Failed to initialize context! Error: ", e.what());
    }
    catch (...) {
      TORCH_CHECK(false, "Vulkan: Failed to initialize context! Error: Unknown");
    }

    return nullptr;
  }());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      context,
      "Invalid Vulkan context!");

  return context.get();
}

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

Descriptor::Set dispatch_prologue(
    Command::Buffer& command_buffer,
    const ShaderLayout::Signature& shader_layout_signature,
    const ShaderSource& shader_descriptor,
    const utils::uvec3& local_work_group_size) {
  Context* const context = api::context();
  Descriptor& descriptor = context->descriptor();
  ShaderLayoutCache& shader_layout_cache = context->shader_layout_cache();
  ShaderCache& shader_cache = context->shader_cache();
  PipelineLayoutCache& pipeline_layout_cache = context->pipeline_layout_cache();
  ComputePipelineCache& pipeline_cache = context->pipeline_cache();

  const VkDescriptorSetLayout shader_layout = shader_layout_cache.retrieve(
      shader_layout_signature);

  const VkPipelineLayout pipeline_layout = \
      pipeline_layout_cache.retrieve(shader_layout);

  const VkPipeline pipeline = pipeline_cache.retrieve({
      pipeline_layout_cache.retrieve(shader_layout),
      shader_cache.retrieve(shader_descriptor),
      local_work_group_size});

  command_buffer.bind(pipeline, pipeline_layout, local_work_group_size);

  return descriptor.pool.allocate(shader_layout, shader_layout_signature);
}

void dispatch_epilogue(
    Command::Buffer& command_buffer,
    const Descriptor::Set& descriptor_set,
    const utils::uvec3& global_work_group) {
  command_buffer.bind(descriptor_set);
  command_buffer.dispatch(global_work_group);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
