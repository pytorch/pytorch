#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor batch_norm(
    const at::Tensor& input_arg,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    bool training,
    double /* momentum, not used in eval mode */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  TORCH_CHECK(!training, "Vulkan batchnorm only supports evaluation mode.");
  TORCH_CHECK(
      weight_opt && weight_opt->defined() && bias_opt && bias_opt->defined(),
      "Vulkan batchnorm expects weight and bias arguments to be defined");
  TORCH_CHECK(
      running_mean_opt && running_mean_opt->defined(),
      "running_mean must be defined in evaluation mode.");
  TORCH_CHECK(
      running_var_opt && running_var_opt->defined(),
      "running_var must be defined in evaluation mode.");
  TORCH_CHECK(input_arg.dim() == 4, "Vulkan batchnorm expects 4-dim input!");
  TORCH_CHECK(
      channels_size(input_arg) % 4 == 0,
      "Vulkan batchnorm expects channel dim to be multiple of 4!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);
  const IntArrayRef v_input_sizes = v_input.sizes();

  auto num_features = v_input.sizes()[1];
  auto channels_ext = num_features / 4;

  const Tensor weight_opt_3d = weight_opt->reshape({num_features, 1, 1});
  const Tensor weight =
      weight_opt_3d.is_vulkan() ? weight_opt_3d : weight_opt_3d.vulkan();
  const vTensor& v_weight = convert(weight);
  TORCH_CHECK(
      weight.numel() == num_features,
      "weight tensor should contain ",
      num_features,
      " elements!");

  const Tensor bias_opt_3d = bias_opt->reshape({num_features, 1, 1});
  const Tensor bias =
      bias_opt_3d.is_vulkan() ? bias_opt_3d : bias_opt_3d.vulkan();
  const vTensor& v_bias = convert(bias);
  TORCH_CHECK(
      bias.numel() == num_features,
      "bias tensor should contain ",
      num_features,
      " elements!");

  const Tensor running_mean_opt_3d =
      running_mean_opt->reshape({num_features, 1, 1});
  const Tensor running_mean = running_mean_opt_3d.is_vulkan()
      ? running_mean_opt_3d
      : running_mean_opt_3d.vulkan();
  const vTensor& v_running_mean = convert(running_mean);
  TORCH_CHECK(
      running_mean.numel() == num_features,
      "running mean tensor should contain ",
      num_features,
      " elements!");

  const Tensor running_var_opt_3d =
      running_var_opt->reshape({num_features, 1, 1});
  const Tensor running_var = running_var_opt_3d.is_vulkan()
      ? running_var_opt_3d
      : running_var_opt_3d.vulkan();
  const vTensor& v_running_var = convert(running_var);
  TORCH_CHECK(
      running_var.numel() == num_features,
      "running var tensor should contain ",
      num_features,
      " elements!");

  api::Context* const context = api::context();

  vTensor v_output{
      context,
      v_input_sizes,
      v_input.options(),
  };

  const struct Block final {
    uvec3 iextents;
    int32_t channels_ext;
    float epsilon;
  } block{
      v_output.extents(),
      safe_downcast<int32_t>(channels_ext),
      safe_downcast<float>(eps)};

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(batchnorm),
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
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_mean.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_var.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::batch_norm"), TORCH_FN(batch_norm));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
