#include <ATen/ATen.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/vulkan/Context.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

//
// Utility functions for memcpy
//

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping) {
  if (src.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping);
  } else if (src.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping);
  } else if (src.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " at::kHalf or at::Float but got ",
        src.dtype());
  }
}

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst) {
  if (dst.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst);
  } else if (dst.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " at::kHalf or at::Float but got ",
        dst.dtype());
  }
}

//
// CPU <-> GPU copy implementations (these functions use Transfer commands)
//

void transfer_cpu_to_vulkan(const Tensor& src, vTensor& v_dst) {
  api::Context* const context = api::context();

  // Convert to dtype corresponding to the image format of the texture to
  // ensure that byte alignment is consistent when copying. In some cases
  // a 16 bit format will be used for at::kFloat.
  Tensor src_nc4hw = utils::nchw_to_nc4hw(src).to(v_dst.texture_dtype());

  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
  // Copy data into the staging buffer
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy_to_mapping(src_nc4hw, mapping);
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}

void transfer_vulkan_to_cpu(vTensor& v_src, Tensor& dst) {
  api::Context* const context = api::context();

  // Temporary tensor to receive copied NC4HW data
  at::Tensor dst_tmp = utils::create_staging_tensor(v_src);

  api::StorageBuffer staging(context, v_src.texture_dtype(), v_src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    api::PipelineBarrier pipeline_barrier{};
    utils::copy_vtensor_to_buffer(
        v_src, staging.buffer(), pipeline_barrier, fence.get_submit_handle());

    fence.wait();

    context->flush();
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    memcpy_from_mapping(mapping, dst_tmp);
  }

  context->fences().return_fence(fence);

  dst = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes()).to(v_src.dtype());
}

static void transfer_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanImage, api::VulkanImage>(
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

//
// CPU <-> GPU copy implementations (these functions use compute shaders)
//

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  // Ensure that src is contiguous in its memory format
  Tensor src_contig = src.contiguous(src.suggest_memory_format());

  // Note that the float data type has been enforced for the storage buffer
  // below. The reason for this is that the nchw_to_image and image_to_nchw
  // shaders which perform the transfer to/from an image texture expect a buffer
  // of floats as input. GLSL/Vulkan does not natively support 16 bit arithmetic
  // types, so for now storage buffers created for compute shaders must define
  // floats as their base data type.
  api::StorageBuffer staging(context, at::kFloat, dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

    // If the dtype() of src is at::kHalf, then first convert it to 32 bit
    // float. This is required since the nchw_to_image shader uses a float
    // buffer as input (note that at::kFloat is used to create the StorageBuffer
    // above).
    if (src.dtype() == at::kHalf) {
      memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
    } else {
      memcpy_to_mapping(src_contig, mapping);
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  TORCH_CHECK(
      !src.is_quantized(),
      "Copy of vulkan quantized tensors to cpu is currently disabled!");
  api::Context* const context = api::context();

  // Refer to the comment in pack_cpu_to_vulkan for why at::kFloat is specified
  // for the storage buffer below.
  api::StorageBuffer staging(context, at::kFloat, src.gpu_numel());

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

    // If the dtype() of dst is at::kHalf, then copy the data into a float
    // version of it first, similar to pack_cpu_to_vulkan().
    if (dst.dtype() == at::kHalf) {
      Tensor dst_float = dst.to(at::kFloat);
      memcpy_from_mapping(mapping, dst_float);
      dst = dst_float.to(at::kHalf);
    } else {
      memcpy_from_mapping(mapping, dst);
    }
  }

  context->fences().return_fence(fence);
}

//
// Copy op implementations
//

Tensor& copy_(Tensor& dst, const Tensor& src) {
  // Check that sizes are equal
  TORCH_CHECK(
      dst.sizes() == src.sizes(), "Vulkan copy_: Tensor sizes are mismatched!");

  // X -> Vulkan
  if (at::kVulkan == dst.device().type()) {
    vTensor& v_self = convert(dst);

    // Vulkan -> Vulkan
    if (at::kVulkan == src.device().type()) {
      vTensor& v_src = convert(src);
      transfer_vulkan_to_vulkan(v_src, v_self);
    }
    // CPU -> Vulkan
    else {
      pack_cpu_to_vulkan(src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (dst.device().is_cpu()) {
      pack_vulkan_to_cpu(v_src, dst);
    } else {
      TORCH_CHECK(false, "Unsupported!");
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return dst;
}

vTensor to_vulkan(at::Tensor& src, const api::StorageType storage_type) {
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan(): input tensor must be a CPU tensor!")

  vTensor v_ret{
      api::context(),
      src.sizes(),
      src.scalar_type(),
      storage_type,
      src.suggest_memory_format(),
  };

  ops::pack_cpu_to_vulkan(src, v_ret);

  return v_ret;
}

at::Tensor from_vulkan(vTensor& v_src) {
  at::TensorOptions opt(at::kCPU);
  opt = opt.dtype(v_src.dtype());

  at::Tensor ret = at::empty(v_src.sizes(), opt).to(v_src.memory_format());
  ops::pack_vulkan_to_cpu(v_src, ret);
  return ret;
}

//
// VulkanImpl
//

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return api::available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
