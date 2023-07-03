#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor softmax_internal(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float,
    const bool log_softmax) {
  TORCH_CHECK(
      input_arg.dim() == 4, "Vulkan softmax expects 4-dimensional input!");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();
  c10::SmallVector<int64_t, 4u> output_sizes{
      v_input_sizes[Layout::Activation4D::batch],
      v_input_sizes[Layout::Activation4D::channels],
      v_input_sizes[Layout::Activation4D::height],
      v_input_sizes[Layout::Activation4D::width],
  };

  vTensor v_output{
      context,
      output_sizes,
      input_arg.scalar_type(),
  };

  // we have custom global workgroup extents for softmax to enable
  // shader algorithms that avoid redundant denominator computations
  const api::utils::uvec3 global_workgroup_extents = v_output.extents();
  api::utils::ivec4 input_shader_extents = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // early_exit is the global workgroup position-based condition for
  // unnecessary invocations to exit.
  api::utils::ivec4 early_exit = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // for batch/height/width, they share the same shader
  // vectorized by input_dim_stride for each dimension case
  api::utils::ivec4 input_dim_stride = {
      0,
      0,
      0,
      0, // zero pad
  };
  api::ShaderInfo shader_descriptor;
  if (log_softmax) {
    if (dim == 1) {
      shader_descriptor = VK_KERNEL(log_softmax);
    } else {
      TORCH_CHECK(
          dim == 1,
          "Vulkan log_softmax expects 4-dimensional input with dim=1!");
    }
  } else {
    if (dim == 1) {
      // for channel dim case, the memory layout forces
      // a different shader algorithm than other dims
      input_shader_extents.data[2u] =
          v_input_sizes[Layout::Activation4D::batch];
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else {
      // for batch, height, width dim case, we can reuse a single shader
      // with vectorized parameters
      if (dim == 0) {
        early_exit.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
        input_dim_stride.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
      } else if (dim == 2) {
        early_exit.data[1u] = 1;
        input_dim_stride.data[1u] = 1;
      } else { // dim == 3
        early_exit.data[0u] = 1;
        input_dim_stride.data[0u] = 1;
      }
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  }

  const struct Block final {
    ivec4 input_shader_extents;
    ivec4 input_tensor_dims;
    ivec4 input_dim_stride;
    ivec4 early_exit;
  } block{
      input_shader_extents,
      {
          safe_downcast<int32_t>(v_input_sizes[Layout::Activation4D::batch]),
          safe_downcast<int32_t>(v_input_sizes[Layout::Activation4D::channels]),
          safe_downcast<int32_t>(v_input_sizes[Layout::Activation4D::height]),
          safe_downcast<int32_t>(v_input_sizes[Layout::Activation4D::width]),
      }, // input_tensor_dims
      input_dim_stride,
      early_exit};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_extents,
      // local work group size
      adaptive_work_group_size(global_workgroup_extents),
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

  return convert(v_output);
}

Tensor softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(input_arg, dim, half_to_float, false);
}

Tensor log_softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(input_arg, dim, half_to_float, true);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_softmax", TORCH_FN(softmax));
  m.impl("_log_softmax", TORCH_FN(log_softmax));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
