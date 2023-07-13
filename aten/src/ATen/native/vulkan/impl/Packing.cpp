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
          case c10::ScalarType::QUInt8:
            return VK_KERNEL(nchw_to_image_uint8);
          case c10::ScalarType::QInt8:
            return VK_KERNEL(nchw_to_image_int8);
          case c10::ScalarType::QInt32:
            return VK_KERNEL(nchw_to_image_int32);
          default:
            TORCH_CHECK(
                false,
                "Vulkan quantization currently not supported for dtype ",
                v_dst.dtype());
        }
      default:
        TORCH_CHECK(false, "No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        TORCH_CHECK(false, "Requested storage type must be a texture type.");
    }
  }

  if (v_dst.dtype() == at::kFloat) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(nchw_to_image2d);
      default:
        TORCH_CHECK(false, "No kernel available!");
    }
  } else if (v_dst.dtype() == at::kBool) {
    switch (v_dst.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(nchw_to_image_bool);
      default:
        TORCH_CHECK(false, "No kernel available!");
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype!");
  }
}

api::ShaderInfo get_image_to_nchw_shader(const vTensor& v_src) {
  if (v_src.is_quantized()) {
    auto plane_size =
        dim_at<Dim4D::Height>(v_src) * dim_at<Dim4D::Width>(v_src);
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        switch (v_src.dtype()) {
          case c10::ScalarType::QUInt8:
            return plane_size % 4 == 0 ? VK_KERNEL(image_to_nchw_quantized_mul4)
                                       : VK_KERNEL(image_to_nchw_uint);
          case c10::ScalarType::QInt8:
            return plane_size % 4 == 0 ? VK_KERNEL(image_to_nchw_quantized_mul4)
                                       : VK_KERNEL(image_to_nchw_uint);
          case c10::ScalarType::QInt32:
            return VK_KERNEL(image_to_nchw_int32);
          default:
            TORCH_CHECK(
                false,
                "Vulkan quantization currently not supported for dtype ",
                v_src.dtype());
        }
      default:
        TORCH_CHECK(false, "No kernel available!");
      case api::StorageType::BUFFER:
      case api::StorageType::UNKNOWN:
        TORCH_CHECK(false, "Requested storage type must be a texture type.");
    }
  }

  if (v_src.dtype() == at::kFloat) {
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw);
      case api::StorageType::TEXTURE_2D:
        return VK_KERNEL(image2d_to_nchw);
      default:
        TORCH_CHECK(false, "No kernel available!");
    }
  } else if (v_src.dtype() == at::kBool) {
    switch (v_src.storage_type()) {
      case api::StorageType::TEXTURE_3D:
        return VK_KERNEL(image_to_nchw_uint);
      default:
        TORCH_CHECK(false, "No kernel available!");
    }
  } else {
    TORCH_CHECK(false, "Unsupported dtype!");
  }
}

struct ToFromTextureParams final {
  api::utils::ivec3 extents;
  int32_t plane_size;
  api::utils::ivec2 c_info;
};

void record_nchw_to_image_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle) {
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

void record_image_to_nchw_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle) {
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

  if (v_src.dtype() == c10::ScalarType::QUInt8 ||
      v_src.dtype() == c10::ScalarType::QInt8 || v_src.dtype() == at::kBool) {
    if (plane_size % 4 == 0) {
      global_size.data[0u] = plane_size / 4;
      global_size.data[1u] = 1;
      local_size.data[0u] *= local_size.data[1u];
      local_size.data[1u] = 1;
    } else {
      uint32_t numel = v_src.numel();
      global_size = {api::utils::div_up(numel, uint32_t(4)), 1u, 1u};
      local_size = {64u, 1u, 1u};
    }
  }

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
    const VkFence fence_handle) {
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

void record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier pipeline_barrier,
    const VkFence fence_handle) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_src.numel());

  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());

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
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

} // namespace packing
} // namespace vulkan
} // namespace native
} // namespace at
