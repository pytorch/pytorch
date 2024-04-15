#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/impl/Packing.h>

namespace at {
namespace native {
namespace vulkan {
namespace packing {

api::ShaderInfo get_nchw_to_image_shader(const vTensor& v_dst) {
  if (v_dst.is_quantized()) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        switch (v_dst.dtype()) {
          case api::ScalarType::QUInt8:
            return VK_KERNEL(nchw_to_image_uint8);
          case api::ScalarType::QInt8:
            return VK_KERNEL(nchw_to_image_int8);
          case api::ScalarType::QInt32:
            return VK_KERNEL(nchw_to_image_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_dst.dtype());
        }
      case api::StorageType::TEXTURE_2D:
        switch (v_dst.dtype()) {
          case api::ScalarType::QUInt8:
            return VK_KERNEL(nchw_to_image2d_uint8);
          case api::ScalarType::QInt8:
            return VK_KERNEL(nchw_to_image2d_int8);
          case api::ScalarType::QInt32:
            return VK_KERNEL(nchw_to_image2d_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_dst.dtype());
        }
      default:
        VK_THROW("No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        VK_THROW("Requested storage type must be a texture type.");
    }
  }

  if (v_dst.dtype() == api::kFloat) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(nchw_to_image2d);
      default:
        VK_THROW("No kernel available!");
    }
  } else if (v_dst.dtype() == api::kBool) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image_bool);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype!");
  }
}

api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src) {
  if (v_src.is_quantized() || v_src.dtype() == api::kBool) {
    auto plane_size =
        dim_at<Dim4D::Height>(v_src) * dim_at<Dim4D::Width>(v_src);
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        switch (v_src.dtype()) {
          case api::ScalarType::QUInt8:
          case api::ScalarType::QInt8:
          case api::kBool:
            return plane_size % 4 == 0 ? VK_KERNEL(image_to_nchw_quantized_mul4)
                                       : VK_KERNEL(image_to_nchw_uint);
          case api::ScalarType::QInt32:
            return VK_KERNEL(image_to_nchw_int32);
          default:
            VK_THROW(
                "Vulkan quantization currently not supported for dtype ",
                v_src.dtype());
        }
      default:
        VK_THROW("No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        VK_THROW("Requested storage type must be a texture type.");
    }
  }

  if (v_src.dtype() == api::kFloat) {
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(image2d_to_nchw);
      default:
        VK_THROW("No kernel available!");
    }
  } else {
    VK_THROW("Unsupported dtype!");
  }
}

struct ToFromTextureParams final {
  api::utils::ivec3 extents;
  int32_t planeSize;
  api::utils::ivec2 channelInfo;
};

void record_nchw_to_image_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  int32_t height =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(v_dst));
  int32_t width =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(v_dst));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(v_dst));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  ToFromTextureParams block{
      api::utils::make_ivec3(v_dst.extents()),
      plane_size,
      {c_depth, channels},
  };

  api::UniformParamsBuffer params(context, block);
  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      // params buffer
      params.buffer());
}

bool record_image_to_nchw_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  api::utils::uvec3 global_size = v_src.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  int32_t height =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Height>(v_src));
  int32_t width =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Width>(v_src));
  int32_t channels =
      api::utils::safe_downcast<int32_t>(dim_at<Dim4D::Channel>(v_src));

  int32_t plane_size = height * width;
  int32_t c_depth = api::utils::div_up(channels, 4);

  ToFromTextureParams block{
      api::utils::make_ivec3(v_src.extents()),
      plane_size,
      {c_depth, channels},
  };

  if (v_src.dtype() == api::ScalarType::QUInt8 ||
      v_src.dtype() == api::ScalarType::QInt8 || v_src.dtype() == api::kBool) {
    // Special case using optimized shader, image_to_nchw_quantized_mul4
    if (plane_size % 4 == 0) {
      global_size.data[0u] = plane_size / 4;
      global_size.data[1u] = 1;
      local_size.data[0u] *= local_size.data[1u];
      local_size.data[1u] = 1;
    }
    // Global and local size for regular 1D buffer.
    else {
      uint32_t numel = v_src.numel();
      global_size = {api::utils::div_up(numel, uint32_t(4)), 1u, 1u};
      local_size = {64u, 1u, 1u};
    }
  }

  api::UniformParamsBuffer params(context, block);
  return context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      dst_buffer,
      // params buffer
      params.buffer());
}

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  uint32_t gpu_buf_len = api::utils::safe_downcast<uint32_t>(v_dst.gpu_numel());

  api::utils::uvec3 global_size = {gpu_buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_dst.get_cpu_buffer_metadata());

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(buffer_to_buffer),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_dst.buffer_metadata(),
      src_buffer,
      cpu_buffer_metadata.buffer());
}

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    VkFence fence_handle) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_src.numel());

  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());

  return context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(buffer_to_buffer),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      fence_handle,
      // shader arguments
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

vTensor channel_image_repacking(
    const vTensor& v_input,
    api::GPUMemoryLayout target_layout,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
      v_input.storage_type(),
      target_layout,
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // The shader assumes a 4d nchw to calculate the lookup coordinate.
  // If the input is not 4d, we need to pad it with 1's on the front.
  const struct Block final {
    api::utils::ivec4 sizes;
  } block{
      api::utils::make_ivec4_prepadded1(v_input.sizes()),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      // VK_KERNEL(packing_channel_to_height),
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return v_output;
}

vTensor convert_image_channels_packed_to_height_packed(const vTensor& v_input) {
  return channel_image_repacking(
      v_input,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      VK_KERNEL(convert_channels_to_height_packed));
}

vTensor convert_image_channels_packed_to_width_packed(const vTensor& v_input) {
  return channel_image_repacking(
      v_input,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      VK_KERNEL(convert_channels_to_width_packed));
}

} // namespace packing
} // namespace vulkan
} // namespace native
} // namespace at
