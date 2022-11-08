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
      querypool_(config_.queryPoolConfig, adapter_p_),
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
    } catch (const c10::Error& e) {
      TORCH_WARN(
          "Pytorch Vulkan Context: Failed to initialize global vulkan context: ",
          e.what());
    } catch (const std::exception& e) {
      TORCH_WARN(
          "Pytorch Vulkan Context: Failed to initialize global vulkan context: ",
          e.what());
    } catch (...) {
      TORCH_WARN(
          "Pytorch Vulkan Context: Failed to initialize global vulkan context!");
    }

    return nullptr;
  }());

  if (!context) {
    TORCH_WARN(
        "Pytorch Vulkan Context: The global context could not be retrieved "
        "because it failed to initialize.");
  }

  return context.get();
}

//
// UniformParamsBuffer
//

namespace {

void memcpy_to_buffer(const VulkanBuffer& src, VulkanBuffer& dst) {
  MemoryMap dst_mapping(dst, MemoryAccessType::WRITE);

  MemoryMap src_mapping(src, api::MemoryAccessType::READ);
  src_mapping.invalidate();

  void* dst_ptr = dst_mapping.template data<void>();
  void* src_ptr = src_mapping.template data<void>();

  memcpy(dst_ptr, src_ptr, src.mem_size());
}

} // namespace

UniformParamsBuffer::UniformParamsBuffer(const UniformParamsBuffer& other)
    : context_p_(other.context_p_), vulkan_buffer_{} {
  if (other.vulkan_buffer_) {
    vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
        other.vulkan_buffer_.mem_size());

    memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
  }
}

UniformParamsBuffer& UniformParamsBuffer::operator=(
    const UniformParamsBuffer& other) {
  if (&other != this) {
    context_p_ = other.context_p_;

    // Move vulkan_buffer_ to another VulkanBuffer for cleanup
    if (vulkan_buffer_) {
      VulkanBuffer temp_buffer(std::move(vulkan_buffer_));
      context_p_->register_buffer_cleanup(temp_buffer);
    }
    // vulkan_buffer_ should now be empty

    if (other.vulkan_buffer_) {
      vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
          other.vulkan_buffer_.mem_size());

      memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
    }
  }

  return *this;
}

//
// VulkanImpl
//

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
