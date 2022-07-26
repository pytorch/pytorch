#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor stack_feature(const TensorList tensors, vTensor& v_output) {
  api::Context* const context = api::context();

  uint32_t num_tensors = tensors.size();
  for (const auto i : c10::irange(v_output.extents().data[2])) {
    const vTensor& v_t0 = convert(
        tensors[4 * i].is_vulkan() ? tensors[4 * i] : tensors[4 * i].vulkan());
    const vTensor& v_t1 = 4 * i + 1 < num_tensors
        ? convert(
              tensors[4 * i + 1].is_vulkan() ? tensors[4 * i + 1]
                                             : tensors[4 * i + 1].vulkan())
        : v_t0;
    const vTensor& v_t2 = 4 * i + 2 < num_tensors
        ? convert(
              tensors[4 * i + 2].is_vulkan() ? tensors[4 * i + 2]
                                             : tensors[4 * i + 2].vulkan())
        : v_t0;
    const vTensor& v_t3 = 4 * i + 3 < num_tensors
        ? convert(
              tensors[4 * i + 3].is_vulkan() ? tensors[4 * i + 3]
                                             : tensors[4 * i + 3].vulkan())
        : v_t0;

    const struct Block final {
      uvec3 size; // output texture size
      uint32_t z; // texel along the channel-batch dimension to copy data to
    } block{v_output.extents(), i};

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(stack_feature),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        v_t0.extents(),
        // local work group size
        adaptive_work_group_size(v_t0.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_t0.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_t1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_t2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_t3.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }

  return convert(v_output);
}

Tensor stack(const at::TensorList tensors, const int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "Vulkan stack expects at least one tensor");
  TORCH_CHECK(dim == 0, "Vulkan stack expects dim = 0");

  at::Tensor tensor = tensors[0];

  for (const auto& t : tensors) {
    TORCH_CHECK(t.dim() == 2, "Vulkan stack expects 2 dimensional inputs");

    for (const auto d : c10::irange(t.dim())) {
      TORCH_CHECK(
          t.size(d) == tensor.size(d),
          "Vulkan stack inputs must have matching sizes");
    }
  }

  uint32_t num_tensors = tensors.size();
  std::vector<int64_t> output_sizes = {
      num_tensors, tensor.size(0), tensor.size(1)};

  vTensor v_output{api::context(), output_sizes, tensor.options()};

  return stack_feature(tensors, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::stack"), TORCH_FN(stack));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
