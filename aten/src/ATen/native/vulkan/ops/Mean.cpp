#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor mean(
    const at::Tensor& input_arg,
    const OptionalIntArrayRef opt_dim,
    const bool keepdim,
    const optional<ScalarType> dtype) {
  TORCH_CHECK(input_arg.dim() == 4, "Vulkan mean expects 4-dimensional input!");

  static const std::unordered_set<int64_t> expected_dims_set({2, 3});
  std::unordered_set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    auto dim = opt_dim.value();
    for (const auto& d : dim) {
      dims_set.insert(utils::normalize(d, 4));
    }
  }

  TORCH_CHECK(
      dims_set == expected_dims_set,
      "Vulkan mean: currently only supports image-wide reduction!");

  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  const IntArrayRef v_input_sizes = v_input.sizes();

  c10::SmallVector<int64_t, 4u> output_sizes{
      v_input_sizes[Layout::Activation4D::batch],
      v_input_sizes[Layout::Activation4D::channels],
  };

  if (keepdim) {
    output_sizes.push_back(1);
    output_sizes.push_back(1);
  }

  vTensor v_output{
      context,
      output_sizes,
      v_input.options(),
  };

  const struct Block final {
    uvec3 extents;
    int32_t range;
    uvec3 iextents;
  } block{
      v_output.extents(),
      safe_downcast<int32_t>(
          v_input_sizes[Layout::Activation4D::width] *
          v_input_sizes[Layout::Activation4D::height]),
      v_input.extents()};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      keepdim ? VK_KERNEL(mean) : VK_KERNEL(mean2d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_input.extents(),
      // local work group size
      adaptive_work_group_size(v_input.extents()),
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
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(mean));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
