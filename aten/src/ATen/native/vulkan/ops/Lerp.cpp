#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void check_inputs_elementwise_op(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input1) == get_dim<Dim4D::Channel>(input2),
      "Vulkan elementwise ops require channel dimension to be equal!");
  if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Channel>(input1) % 4 == 0,
        "Vulkan elementwise ops require channel to be a multiple of 4 to broadcast along batch dimension!")
  }

  const uint32_t input1_h = get_dim<Dim4D::Height>(input1);
  const uint32_t input1_w = get_dim<Dim4D::Width>(input1);
  const uint32_t input2_h = get_dim<Dim4D::Height>(input2);
  const uint32_t input2_w = get_dim<Dim4D::Width>(input2);

  const std::string broadcast_error_msg =
      "Incompatible input dimensions for broadcasting for Vulkan elementwise op!";
  if (input1_h != input2_h) {
    if (input1_h > input2_h) {
      TORCH_CHECK(input2_h == 1, broadcast_error_msg);
      TORCH_CHECK(input2_w == input1_w || input2_w == 1, broadcast_error_msg);
    } else if (input2_h > input1_h) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
      TORCH_CHECK(input1_w == input2_w || input1_w == 1, broadcast_error_msg);
    }
  } else if (input1_w != input2_w) {
    if (input1_w > input2_w) {
      TORCH_CHECK(input2_w == 1, broadcast_error_msg);
    } else if (input2_w > input1_w) {
      TORCH_CHECK(input1_h == 1, broadcast_error_msg);
    }
  }
}

Tensor _lerp_scalar(
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Scalar& weight_arg) {
  check_inputs_elementwise_op(start_arg, end_arg);
  api::Context* const context = api::context();

  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  vTensor v_output{
      context,
      v_start.sizes(),
      v_start.options(),
  };

  const float weight = weight_arg.to<float>();
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    float weight;
  } block{
      v_output.extents(),
      0u,
      v_start.extents(),
      0u,
      v_end.extents(),
      weight,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(lerp_scalar),
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
      v_start.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& _lerp_scalar_(
    Tensor& self_arg,
    const Tensor& end_arg,
    const Scalar& weight_arg) {
  check_inputs_elementwise_op(self_arg, end_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  const float weight = weight_arg.to<float>();
  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input_extents;
    float alpha;
  } block{
      v_self.extents(),
      0u,
      v_end.extents(),
      weight,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(lerp_scalar_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor _lerp_tensor(
    const Tensor& start_arg,
    const Tensor& end_arg,
    const Tensor& weight_arg) {
  check_inputs_elementwise_op(start_arg, end_arg);
  check_inputs_elementwise_op(start_arg, weight_arg);

  api::Context* const context = api::context();

  const Tensor start = start_arg.is_vulkan() ? start_arg : start_arg.vulkan();
  const vTensor& v_start = convert(start);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end);

  const Tensor weight =
      weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  const vTensor& v_weight = convert(weight_arg);

  vTensor v_output{
      context,
      v_start.sizes(),
      v_start.options(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    uint32_t fill_2;
    uvec3 input3_extents;
    uint32_t fill_3;
  } block{
      v_output.extents(),
      0u,
      v_start.extents(),
      0u,
      v_end.extents(),
      0u,
      v_weight.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(lerp),
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
      v_start.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& _lerp_tensor_(
    Tensor& self_arg,
    const Tensor& end_arg,
    const Tensor& weight_arg) {
  check_inputs_elementwise_op(self_arg, end_arg);
  check_inputs_elementwise_op(self_arg, weight_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const Tensor end = end_arg.is_vulkan() ? end_arg : end_arg.vulkan();
  const vTensor& v_end = convert(end_arg);

  const Tensor weight =
      weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  const vTensor& v_weight = convert(weight_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t fill_0;
    uvec3 input1_extents;
    uint32_t fill_1;
    uvec3 input2_extents;
    uint32_t fill_2;
  } block{
      v_self.extents(),
      0u,
      v_end.extents(),
      0u,
      v_weight.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(lerp_),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      v_end.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor lerp_scalar(
    const Tensor& start,
    const Tensor& end,
    const Scalar& weight) {
  return _lerp_scalar(start, end, weight);
}

Tensor& lerp_scalar_(Tensor& self, const Tensor& end, const Scalar& weight) {
  return _lerp_scalar_(self, end, weight);
}

Tensor lerp_tensor(
    const Tensor& start,
    const Tensor& end,
    const Tensor& weight) {
  if (weight.sizes().size() == 0) {
    return _lerp_scalar(start, end, weight.item<float>());
  }
  return _lerp_tensor(start, end, weight);
}

Tensor& lerp_tensor_(Tensor& self, const Tensor& end, const Tensor& weight) {
  if (weight.sizes().size() == 0) {
    return _lerp_scalar_(self, end, weight.item<float>());
  }
  return _lerp_tensor_(self, end, weight);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Scalar"), TORCH_FN(lerp_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Scalar"), TORCH_FN(lerp_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp.Tensor"), TORCH_FN(lerp_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::lerp_.Tensor"), TORCH_FN(lerp_tensor_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
