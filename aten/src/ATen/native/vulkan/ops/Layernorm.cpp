#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/native_layer_norm.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor layer_norm(
    const at::Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  // We invoke native_layer_norm which returns a tuple of tensors: <layer_norm,
  // mean, 1/sqrt(var+eps)>, but we only need the first tensor (layer_norm).
  std::tuple<Tensor, Tensor, Tensor> native_layer_norm_output =
      at::native_layer_norm(input, normalized_shape, weight_opt, bias_opt, eps);
  return std::get<0>(native_layer_norm_output);
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
