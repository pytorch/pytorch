#include <ATen/native/vulkan/api/OpProfiler.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

void copy_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::Command::Pool& command_pool = context->command().pool;
  api::Command::Buffer& command_buffer = command_pool.stream();
  {
    command_buffer.copy_image(
        // source image
        src.image(
            command_buffer,
            api::PipelineStage::Transfer),
        // destination image
        dst.image(
            command_buffer,
            api::PipelineStage::Transfer,
            api::MemoryAccessType::WRITE),
        // offsets
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        // copy range
        src.extents());
  }
  command_pool.submit(context->gpu().queue, command_buffer);
}

void copy_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::StagingBuffer staging(context, dst.buffer_bytes());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    float* data_ptr = mapping.template data<float>();

    memcpy(
      data_ptr,
      src.contiguous().data_ptr<float>(),
      std::min(src.nbytes(), src.nbytes()));
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void copy_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  api::Context* const context = api::context();

  api::StagingBuffer staging(context, src.buffer_bytes());

  api::VulkanFence fence = context->fences().get_fence();
  utils::pack_vtensor_to_staging(
      src, staging.buffer(), fence.get_submit_handle());

  fence.wait();
  context->fences().return_fence(fence);

  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    float* data_ptr = mapping.template data<float>();

    memcpy(
        dst.data_ptr<float>(),
        data_ptr,
        std::min(src.nbytes(), dst.nbytes()));
  }
}

Tensor& copy_(Tensor& self, const Tensor& src) {
  // Check that sizes are equal
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "Vulkan copy_: Tensor sizes are mismatched!");

  // X -> Vulkan
  if (at::kVulkan == self.device().type()) {
    vTensor& v_self = convert(self);

    // Vulkan -> Vulkan
    if (at::kVulkan == src.device().type()) {
      vTensor& v_src = convert(src);
      copy_vulkan_to_vulkan(v_src, v_self);
    }
    // CPU -> Vulkan
    else {
      copy_cpu_to_vulkan(src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (self.device().is_cpu()) {
      copy_vulkan_to_cpu(v_src, self);
    }
    else {
      TORCH_CHECK(false, "Unsupported!");
    }

    // Transfer back to CPU is interpreted as a sync point.
    // Flushes GPU resources.
    api::Context* const context = api::context();
    context->flush();
  }
  else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return self;
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
