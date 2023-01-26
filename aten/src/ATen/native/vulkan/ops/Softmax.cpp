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
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      input_arg.dim() == 4, "Vulkan softmax expects 4-dimensional input!");

  TORCH_CHECK(dim == 1, "Vulkan softmax expects dim == 1 (channel)");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  TORCH_CHECK(
      v_input_sizes[Layout::Activation4D::batch] == 1,
      "Vulkan softmax expects batch dim == 1");

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

  const api::utils::uvec3 global_work_group_size = {
      safe_downcast<uint32_t>(v_input_sizes[Layout::Activation4D::width]),
      safe_downcast<uint32_t>(v_input_sizes[Layout::Activation4D::height]),
      1,
  };
  const api::utils::uvec3 local_work_group_size = {8, 8, 1};

  const struct Block final {
    uvec3 iextents;
    int last_texel_end_offset;
  } block{
      v_input.extents(),
      safe_downcast<int32_t>(
          (v_input_sizes[Layout::Activation4D::channels] - 1) % 4)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_work_group_size,
      // local work group size
      local_work_group_size,
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
  return softmax_internal(input_arg, dim, half_to_float, VK_KERNEL(softmax));
}

Tensor log_softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(
      input_arg, dim, half_to_float, VK_KERNEL(log_softmax));
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
