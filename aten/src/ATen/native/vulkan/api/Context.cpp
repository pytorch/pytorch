#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/vulkan/Context.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {

Context::Context(const VkInstance instance, size_t adapter_i)
    : instance_(instance),
      adapter_p_(runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      threadcontext_(gpu()) {
}

Context::~Context() {
  // Let the device know the context is done with the queue
  adapter_p_->return_queue(queue_);
  // Do not call flush() since all per-thread objects will be destroyed as each thread exits
}

void Context::flush() {
  VK_CHECK(vkQueueWaitIdle(queue()));

  resource().pool.purge();
  descriptor().pool.purge();
  command().pool.purge();
}

void Context::wait(const at::Tensor& src) {
  // wait only if Vulkan tensor
  if (at::kVulkan == src.device().type()) {
    api::Command::Pool& command_pool = command().pool;
    api::Command::Buffer& command_buffer = command_pool.stream();

    using Future = ops::vTensor::Future<const void, ops::vTensor::Access::Read>;
    const ops::vTensor& v_src = ops::convert(src);
    const Future v_src_future = v_src.host<const void>(command_buffer);

    // This wait() is a no-op if data is not out of sync.  More often than
    // not though, waits here are expected as the GPU catches up with
    // compute submitted from CPU.
    v_src_future.wait();
  }
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      return new Context(runtime()->instance(), runtime()->default_adapter_i());
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
