#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor glu(const at::Tensor& input_arg, const int64_t dim = -1) {
  TORCH_CHECK(input_arg.dim() == 4, "Vulkan glu only supports 4-dim input!");
  TORCH_CHECK(
      dim == 1,
      "Vulkan glu only supports GLU for dim = 1, but got dim = ",
      dim);
  // For now, only allow if channels dim is a multiple of 4
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input_arg) % 4 == 0,
      "Vulkan glu expects channel dim to be multiple of 4!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  auto output_ch_size = v_input.sizes()[1] / 2;

  api::Context* const context = api::context();

  vTensor v_output{
      context,
      {v_input_sizes[0], output_ch_size, v_input_sizes[2], v_input_sizes[3]},
      v_input.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    int32_t chext;
  } block{v_output.extents(), safe_downcast<int32_t>(output_ch_size)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(glu_channel_mul4),
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

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::glu"), TORCH_FN(glu));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
