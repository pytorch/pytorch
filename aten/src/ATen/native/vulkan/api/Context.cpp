#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/vulkan/Context.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Context::Context(size_t adapter_i, const ContextConfig& config)
    : config_(config),
      // Important handles
      adapter_p_(runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      // Resource pools
      command_pool_(device_, queue_.family_index, config_.cmdPoolConfig),
      descriptor_pool_(device_, config_.descriptorPoolConfig),
      fences_(device_),
// Diagnostics
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
      querypool_(device_, config_.queryPoolConfig),
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
      // Command buffer submission
      cmd_mutex_{},
      cmd_(VK_NULL_HANDLE),
      submit_count_{0u},
      // Memory Management
      buffer_clearlist_mutex_{},
      buffers_to_clear_{},
      image_clearlist_mutex_{},
      images_to_clear_{} {
}

Context::~Context() {
  flush();
  // Let the device know the context is done with the queue
  adapter_p_->return_queue(queue_);
}

DescriptorSet Context::submit_compute_prologue(
    CommandBuffer& command_buffer,
    const ShaderSource& shader_descriptor,
    const utils::uvec3& local_workgroup_size) {
  const VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  const VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  const VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout_cache().retrieve(shader_layout),
       shader_cache().retrieve(shader_descriptor),
       local_workgroup_size});

  command_buffer.bind_pipeline(pipeline, pipeline_layout, local_workgroup_size);

  return descriptor_pool().get_descriptor_set(
      shader_layout, shader_descriptor.kernel_layout);
}

void Context::submit_compute_epilogue(
    CommandBuffer& command_buffer,
    const DescriptorSet& descriptors,
    const PipelineBarrier& pipeline_barrier,
    const utils::uvec3& global_workgroup_size) {
  command_buffer.bind_descriptors(descriptors.get_bind_handle());
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
  // Serialize recording to the shared command buffer. Do not initialize with a
  // mutex just yet, since in some cases it will be externally managed.
  std::unique_lock<std::mutex> cmd_lock;
  // Refer to comments in submit_compute_job for explanation.
  if (fence_handle == VK_NULL_HANDLE) {
    cmd_lock = std::unique_lock<std::mutex>(cmd_mutex_);
  }

  set_cmd();

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  uint32_t log_idx = querypool_.shader_profile_begin(
      cmd_,
      "copy_texture_to_texture",
      create_extent3d({0, 0, 0}),
      create_extent3d({0, 0, 0}));
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

  cmd_.insert_barrier(pipeline_barrier);

  cmd_.copy_texture_to_texture(
      source, destination, copy_range, src_offset, dst_offset);

#ifdef USE_VULKAN_GPU_DIAGNOSTICS
  querypool_.shader_profile_end(cmd_, log_idx);
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */

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
  descriptor_pool_.flush();

  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  buffers_to_clear_.clear();
  images_to_clear_.clear();
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const uint32_t submit_frequency = 16u;

      const CommandPoolConfig cmd_config{
          32u, // cmdPoolInitialSize
          8u, // cmdPoolBatchSize
      };

      const DescriptorPoolConfig descriptor_pool_config{
          1024u, // descriptorPoolMaxSets
          1024u, // descriptorUniformBufferCount
          1024u, // descriptorStorageBufferCount
          1024u, // descriptorCombinedSamplerCount
          1024u, // descriptorStorageImageCount
          32u, // descriptorPileSizes
      };

      const QueryPoolConfig query_pool_config{
          4096u, // maxQueryCount
          256u, // initialReserveSize
      };

      const ContextConfig config{
          submit_frequency, // cmdSubmitFrequency
          cmd_config, // cmdPoolConfig
          descriptor_pool_config, // descriptorPoolConfig
          query_pool_config, // queryPoolConfig
      };

      return new Context(runtime()->default_adapter_i(), config);
    } catch (const std::exception& e) {
      TORCH_CHECK(
          false, "Vulkan: Failed to initialize context! Error: ", e.what());
    } catch (...) {
      TORCH_CHECK(
          false, "Vulkan: Failed to initialize context! Error: Unknown");
    }

    return nullptr;
  }());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(context, "Invalid Vulkan context!");

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

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
