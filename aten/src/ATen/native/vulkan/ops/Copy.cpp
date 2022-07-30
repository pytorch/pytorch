#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

void copy_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::PipelineBarrier pipeline_barrier{};

  context->submit_texture_copy(
      // pipeline barrier
      pipeline_barrier,
      // images
      src.image(pipeline_barrier, api::PipelineStage::TRANSFER),
      dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

void copy_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::StagingBuffer staging(context, dst.buffer_bytes());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    if (src.dtype() == c10::kQUInt8) {
      c10::quint8* data_ptr = mapping.template data<c10::quint8>();
      memcpy(
          data_ptr,
          src.contiguous().data_ptr<c10::quint8>(),
          std::min(src.nbytes(), src.nbytes()));
    } else {
      float* data_ptr = mapping.template data<float>();
      memcpy(
          data_ptr,
          src.contiguous().data_ptr<float>(),
          std::min(src.nbytes(), src.nbytes()));
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void copy_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  api::Context* const context = api::context();

  api::StagingBuffer staging(context, src.buffer_bytes());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    utils::pack_vtensor_to_staging(
        src, staging.buffer(), fence.get_submit_handle());

    fence.wait();

    context->flush();
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    if (dst.is_quantized()) {
      c10::quint8* data_ptr = mapping.template data<c10::quint8>();
      memcpy(
          dst.data_ptr<c10::quint8>(),
          data_ptr,
          std::min(src.nbytes(), dst.nbytes()));
    } else {
      float* data_ptr = mapping.template data<float>();
      memcpy(
          dst.data_ptr<float>(),
          data_ptr,
          std::min(src.nbytes(), dst.nbytes()));
    }
  }

  context->fences().return_fence(fence);
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
      TORCH_CHECK(
          src.dtype() == c10::kQUInt8 || src.dtype() == at::kFloat,
          "Invalid Data Type: expected QUint8 or Float but got ",
          src.dtype());
      copy_cpu_to_vulkan(src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (self.device().is_cpu()) {
      TORCH_CHECK(
          self.dtype() == c10::kQUInt8 || self.dtype() == at::kFloat,
          "Invalid Data Type: expected QUint8 or Float but got ",
          self.dtype());
      copy_vulkan_to_cpu(v_src, self);
    } else {
      TORCH_CHECK(false, "Unsupported!");
    }
  } else {
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
