#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor cumsum(
    const at::Tensor& input_arg,
    const int64_t dim,
    const c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      input_arg.dim() <= 4, "Vulkan cumsum expects input dimension <= 4!");

  TORCH_CHECK(
      get_dim<Dim4D::Batch>(input_arg) == 1,
      "Vulkan cumsum expects batch size <= 1!");

  TORCH_CHECK(dim < 4, "Vulkan cumsum expects dim < 4!");

  if (dim <= 1) {
    // TODO: dim<0, dim=0, dim=1(z axis)
    TORCH_CHECK(false, "Not implemented!");
  }

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      input_arg.sizes(),
      input_arg.options(),
  };

  const struct Block final {
    int32_t axis;
  } block{
      (3 - safe_downcast<int32_t>(dim)),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(cumsum),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_input.extents(),
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
  m.impl(TORCH_SELECTIVE_NAME("aten::cumsum"), TORCH_FN(cumsum));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
