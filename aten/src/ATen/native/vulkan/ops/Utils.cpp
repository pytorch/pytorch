#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

/*
 * This function formats an input tensor in NCHW layout to NC4HW layout such
 * that the buffer of the formatted tensor can be directly copied into a GPU
 * texture. Conceptually, the formatting can be achieved via the following
 * steps:
 *
 * 1. Given that the src tensor has size {N,C,H,W}
 *
 * 2. Combine the batch and channel dims by reshaping to {N*C, H, W}
 *
 * 3. Determine the amount of padding to add: determine how many channels to add
 *    in order to align N*C to the next multiple of 4
 *
 * 4. Add padding to the tensor so that the batch-channel dimension is a
 *    multiple of four; the shape of the tensor is now {NC_aligned, H, W}
 *
 * 5. Split the batch-channel dimension into groups of 4 by reshaping the tensor
 *    to size {NC_aligned/4, 4, H, W}
 *
 * 6. The groups of 4 channels (dim 1) should be contiguous. Therefore, permute
 *    the dims of the tensor in the order {0, 2, 3, 1}
 *
 * 7. Finally, return a contiguous version of the tensor. The final shape of the
 *    tensor would be {NC_aligned/4, H, W, 4}
 */
Tensor nchw_to_nc4hw(const Tensor& src) {
  uint32_t N = get_dim<Dim4D::Batch>(src.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(src.sizes());
  uint32_t H = get_dim<Dim4D::Height>(src.sizes());
  uint32_t W = get_dim<Dim4D::Width>(src.sizes());

  uint32_t C_aligned = api::utils::align_up(C, 4u);
  uint32_t NC4 = (N * C_aligned) / 4;

  // Add padding to the tensor so that the channel dim is a multiple of 4
  Tensor padding = at::zeros({N, C_aligned - C, H, W}, src.options());
  Tensor src_padded = at::cat({src.reshape({N, C, H, W}), padding}, 1);
  // Reshape to group channels into groups of 4 and permute so that the groups
  // are in the first dimension so that they are contiguous
  Tensor src_NC4HW = src_padded.reshape({NC4, 4, H, W}).permute({0, 2, 3, 1});

  // Return a contiguous version of the tensor
  return src_NC4HW.contiguous();
}

/*
 * Creates a staging tensor into which texture data, which will be in NC4HW
 * format, can be copied directly. The shape of the staging tensor will be the
 * same as the tensor produced by a call to format_src_tensor().
 */
Tensor create_staging_tensor(const vTensor& v_in) {
  uint32_t N = get_dim<Dim4D::Batch>(v_in.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(v_in.sizes());
  uint32_t H = get_dim<Dim4D::Height>(v_in.sizes());
  uint32_t W = get_dim<Dim4D::Width>(v_in.sizes());

  uint32_t NC4 = N * api::utils::div_up(C, 4u);

  // Note that the dtype corresponding with the texture format of the vTensor is
  // used instead of options().dtype(). This is to ensure the number of bytes in
  // the staging tensor matches the number of bytes in the image texture. Refer
  // to comments for api::vk_format()
  return at::empty(
      {NC4, H, W, 4}, at::device(at::kCPU).dtype(v_in.texture_dtype()));
}

/*
 * After copying texture data, which will be in NC4HW format, to a staging
 * tensor created in create_staging_tensor(), this function reformats the tensor
 * to NCHW format. It essentially reverses the transformations made by
 * format_src_tensor().
 *
 * Note that the sizes of the original tensor must be passed in to fully restore
 * the properties of the original tensor.
 */
Tensor nc4hw_to_nchw(const Tensor& t_in, IntArrayRef sizes) {
  uint32_t N = get_dim<Dim4D::Batch>(sizes);
  uint32_t C = get_dim<Dim4D::Channel>(sizes);
  uint32_t H = get_dim<Dim4D::Height>(sizes);
  uint32_t W = get_dim<Dim4D::Width>(sizes);

  uint32_t C_aligned = api::utils::align_up(C, 4u);

  // Undo the permute step and channel grouping step
  Tensor t_in_padded = t_in.permute({0, 3, 1, 2}).reshape({N, C_aligned, H, W});
  // Remove the padding channels
  Tensor t_in_shaved =
      at::narrow(t_in_padded, /*dim=*/1, /*start*/ 0, /*end*/ C);

  // Reshape to original sizing and dtype and return a contiguous Tensor
  return t_in_shaved.reshape(sizes).contiguous();
}

void copy_buffer_to_vtensor(
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      src_buffer.mem_size() == v_dst.gpu_nbytes(),
      "Vulkan copy_buffer_to_vtensor: source buffer and destination texture "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanBuffer, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src_buffer,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      v_dst.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle) {
  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src.buffer(),
      dst.buffer(),
      // copy details
      {static_cast<uint32_t>(src.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void copy_vtensor_to_buffer(
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    const VkFence fence_handle) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      v_src.gpu_nbytes() == dst_buffer.mem_size(),
      "Vulkan copy_vtensor_to_buffer: source texture and destination buffer "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanImage, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      dst_buffer,
      // copy details
      v_src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer& buffer,
    vTensor& v_self,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    packing::record_nchw_to_buffer_op(
        context, buffer, v_self, pipeline_barrier, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_nchw_to_image_shader(v_self);
    packing::record_nchw_to_image_op(
        context,
        compute_shader,
        buffer,
        v_self,
        pipeline_barrier,
        VK_NULL_HANDLE);
  }
}

void pack_staging_to_vtensor(api::VulkanBuffer& staging, vTensor& v_self) {
  api::PipelineBarrier pipeline_barrier{};
  pack_buffer_to_vtensor(staging, v_self, pipeline_barrier);
}

void pack_vtensor_to_staging(
    vTensor& v_self,
    api::VulkanBuffer& staging,
    const VkFence fence_handle) {
  api::Context* const context = api::context();
  api::PipelineBarrier pipeline_barrier{};

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    packing::record_buffer_to_nchw_op(
        context, v_self, staging, pipeline_barrier, fence_handle);
  } else {
    api::ShaderInfo compute_shader = packing::get_image_to_nchw_shader(v_self);
    packing::record_image_to_nchw_op(
        context,
        compute_shader,
        v_self,
        staging,
        pipeline_barrier,
        fence_handle);
  }
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
