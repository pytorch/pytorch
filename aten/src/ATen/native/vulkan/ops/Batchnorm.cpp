#include <ATen/Context.h>
#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace batchnorm {

struct Params final {
  api::utils::ivec3 out_extents;
  int32_t c4;
  float eps;
};

void record_op(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const vTensor& v_running_mean,
    const vTensor& v_running_var,
    const float eps) {
  api::PipelineBarrier pipeline_barrier{};

  api::utils::uvec3 global_size = v_output.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  uint32_t num_features = get_dim<Dim4D::Channel>(v_input.sizes());
  uint32_t channels_ext = api::utils::div_up(num_features, 4u);

  Params block{
      api::utils::make_ivec3(v_output.extents()),
      api::utils::safe_downcast<int32_t>(channels_ext),
      eps,
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(batchnorm),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
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
}

} // namespace batchnorm

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
  TORCH_CHECK(!training, "Only evaluation mode is supported!");
  TORCH_CHECK(input_arg.dim() == 4, "Input must have dim == 4!");
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input_arg) % 4 == 0,
      "Input must have channels divisible by 4!");

  return run_batchnorm_context(
      input_arg,
      c10::make_intrusive<BatchNormPackedContext>(BatchNormPackedContext(
          weight_opt, bias_opt, running_mean_opt, running_var_opt, eps)));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::batch_norm"), TORCH_FN(batch_norm));
}

#endif /* USE_VULKAN_API */

} // namespace

BatchNormPackedContext::BatchNormPackedContext(
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double eps)
    : unpacked_{c10::AnyType::get()} {
  packed_.reserve(ListArgs::kNumArgs);

  // Each optional tensor arg, if provided should be a 1 dimensional tensor. To
  // achieve more efficient packing as a texture, they are first reshaped to {N,
  // 1, 1}. Eventually this rearrangement should happen automatically in vTensor
  // itself.

  // Weight
  TORCH_CHECK(weight_opt, "Weight must be provided!");
  TORCH_CHECK(weight_opt->dim() == 1, "Weight must have ndim == 1!");

  const int64_t num_features =
      api::utils::safe_downcast<int64_t>(weight_opt->numel());
  const Tensor weight_3d = weight_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(weight_3d.vulkan());

  // Bias
  TORCH_CHECK(bias_opt, "Bias must be provided!");
  TORCH_CHECK(bias_opt->dim() == 1, "Bias must have ndim == 1!");
  TORCH_CHECK(
      bias_opt->numel() == num_features,
      "Bias must have the same numel as weight!");

  const Tensor bias_3d = bias_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(bias_3d.vulkan());

  // Running Mean
  TORCH_CHECK(running_mean_opt, "Running mean must be provided!");
  TORCH_CHECK(running_mean_opt->dim() == 1, "Running mean must have ndim == 1");
  TORCH_CHECK(
      running_mean_opt->numel() == num_features,
      "Running mean must have the same numel as weight!");

  const Tensor running_mean_3d =
      running_mean_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(running_mean_3d.vulkan());

  // Running var
  TORCH_CHECK(running_var_opt, "Running var must be provided!");
  TORCH_CHECK(running_var_opt->dim() == 1, "Running var must have ndim == 1");
  TORCH_CHECK(
      running_var_opt->numel() == num_features,
      "Running var must have the same numel as weight!");

  const Tensor running_var_3d = running_var_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(running_var_3d.vulkan());

  // Epsilon
  packed_.emplace_back(eps);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(ListArgs::kNumArgs);
    unpacked_.emplace_back(weight_opt);
    unpacked_.emplace_back(bias_opt);
    unpacked_.emplace_back(running_mean_opt);
    unpacked_.emplace_back(running_var_opt);
    unpacked_.emplace_back(eps);
  }
}

BatchNormPackedContext BatchNormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return BatchNormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      get_optional_tensor(unpacked, ListArgs::kRunningMean),
      get_optional_tensor(unpacked, ListArgs::kRunningVar),
      unpacked.get(ListArgs::kEps).toDouble());
}

c10::intrusive_ptr<BatchNormPackedContext> create_batchnorm_context(
    c10::optional<Tensor>&& weight_opt,
    c10::optional<Tensor>&& bias_opt,
    c10::optional<Tensor>&& running_mean_opt,
    c10::optional<Tensor>&& running_var_opt,
    bool training,
    double /* momentum */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  return c10::make_intrusive<BatchNormPackedContext>(BatchNormPackedContext(
      weight_opt, bias_opt, running_mean_opt, running_var_opt, eps));
}

Tensor run_batchnorm_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<BatchNormPackedContext>& batchnorm_context) {
  api::Context* const context = api::context();

  const vTensor& v_input = convert(input_arg);

  const vTensor& v_weight = convert(
      batchnorm_context->get_val(BatchNormPackedContext::ListArgs::kWeight)
          .toTensor());

  const vTensor& v_bias = convert(
      batchnorm_context->get_val(BatchNormPackedContext::ListArgs::kBias)
          .toTensor());

  const vTensor& v_running_mean = convert(
      batchnorm_context->get_val(BatchNormPackedContext::ListArgs::kRunningMean)
          .toTensor());

  const vTensor& v_running_var = convert(
      batchnorm_context->get_val(BatchNormPackedContext::ListArgs::kRunningVar)
          .toTensor());

  const float eps = api::utils::safe_downcast<float>(
      batchnorm_context->get_val(BatchNormPackedContext::ListArgs::kEps)
          .toDouble());

  vTensor v_output{
      context,
      v_input.sizes(),
      input_arg.scalar_type(),
  };

  batchnorm::record_op(
      context,
      v_output,
      v_input,
      v_weight,
      v_bias,
      v_running_mean,
      v_running_var,
      eps);

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
