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
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan layernorm expects input of 2d, 3d or 4d!");

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  const Tensor weight =
      weight_opt->is_vulkan() ? *weight_opt : weight_opt->vulkan();

  const Tensor bias = bias_opt->is_vulkan() ? *bias_opt : bias_opt->vulkan();

  std::vector<int64_t> dims_to_reduce;
  for (const auto i : c10::irange(normalized_shape.size())) {
    dims_to_reduce.push_back(input_arg.dim() - i - 1);
  }
  IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  // use the formular in this page to compute the output:
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  // Note that the standard-deviation is calculated via the biased estimator,
  // equivalent to torch.var(input, unbiased=False).
  bool mean_keep_dim = true;
  bool var_unbiasd = false;
  bool var_keep_dim = true;
  auto mean = input.mean(dims_to_reduce_ref, mean_keep_dim);
  auto var = input.var(dims_to_reduce_ref, var_unbiasd, var_keep_dim);
  return input.sub(mean).mul(var.add(eps).pow(-0.5f)).mul(weight).add(bias);
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
