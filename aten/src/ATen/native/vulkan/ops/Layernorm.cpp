#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

void _check_layer_norm_inputs(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight /* optional */,
    const c10::optional<Tensor>& bias /* optional */) {
  const auto normalized_ndim = normalized_shape.size();
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !weight->defined() || weight->sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight->sizes(),
      " and normalized_shape = ",
      normalized_shape);
  TORCH_CHECK(
      !bias->defined() || bias->sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias->sizes(),
      " and normalized_shape = ",
      normalized_shape);

  const auto input_shape = input.sizes();
  const auto input_ndim = input.sizes().size();

  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }
}

Tensor layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  _check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      input_arg.dim() == 3 || input_arg.dim() == 4,
      "Vulkan layernorm expects 3-dim or 4-dim input!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  TORCH_CHECK(
      input_arg.dim() == 3 || v_input_sizes[Layout::Activation4D::batch] == 1,
      "Vulkan layernorm expects batch dim == 1 when the input is 4-dimensional!");

  TORCH_CHECK(
      normalized_shape.size() == 3,
      "Vulkan layernorm expects normalized_shape to have length 3, i.e. [C, H, W]");

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  const auto volume =
      c10::multiply_integers(v_input_sizes.cbegin(), v_input_sizes.end());

  const Tensor weight =
      weight_opt->is_vulkan() ? *weight_opt : weight_opt->vulkan();
  const vTensor& v_weight = convert(weight);

  const Tensor bias = bias_opt->is_vulkan() ? *bias_opt : bias_opt->vulkan();
  const vTensor& v_bias = convert(bias);

  api::Context* const context = api::context();

  vTensor v_output{
      context,
      v_input_sizes,
      v_input.options(),
  };

  const struct Block final {
    uvec3 iextents;
    int32_t volume;
    int32_t last_texel_end_offset;
    float epsilon;
  } block{
      v_input.extents(),
      safe_downcast<int32_t>(volume),
      safe_downcast<int32_t>((v_input_sizes[input_arg.dim() - 3] - 1) % 4),
      safe_downcast<float>(eps)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(layernorm),
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
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::layer_norm"), TORCH_FN(layer_norm));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
