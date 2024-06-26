#include <ATen/native/vulkan/ops/Common.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor pad2d(
    const Tensor& self_arg,
    IntArrayRef padding,
    const api::ShaderInfo& shader_descriptor) {
  const int pad_dim = padding.size();
  const IntArrayRef input_size = self_arg.sizes();
  const int input_dim = input_size.size();

  TORCH_CHECK(
      pad_dim == 1 || pad_dim == 4,
      "Padding sizes must be a 1-tuple or 4-tuple!");
  TORCH_CHECK(input_dim >= 2, "Input tensor must have dim >= 2!");

  api::Context* const context = api::context();

  int pad_left = padding[0];
  int pad_right = padding[0];
  int pad_top = padding[0];
  int pad_bottom = padding[0];
  if (pad_dim == 4) {
    pad_right = padding[1];
    pad_top = padding[2];
    pad_bottom = padding[3];
  }

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  std::vector<int64_t> output_size(input_dim);
  for (const auto d : c10::irange(input_dim)) {
    if (d == input_dim - 1) {
      output_size[d] = input_size[d] + pad_right + pad_left;
    } else if (d == input_dim - 2) {
      output_size[d] = input_size[d] + pad_top + pad_bottom;
    } else {
      output_size[d] = input_size[d];
    }
  }

  vTensor v_output{
      context,
      output_size,
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    uvec4 padding;
  } block{
      v_output.extents(),
      0u,
      {safe_downcast<uint32_t>(pad_left),
       safe_downcast<uint32_t>(pad_right),
       safe_downcast<uint32_t>(pad_top),
       safe_downcast<uint32_t>(pad_bottom)},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor reflection_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(reflection_pad2d));
}

Tensor replication_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(replication_pad2d));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::reflection_pad2d"),
      TORCH_FN(reflection_pad2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::replication_pad2d"),
      TORCH_FN(replication_pad2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
