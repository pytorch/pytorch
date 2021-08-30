#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/custom_class.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {



TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::linear_prepack(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.LinearOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::linear_run(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearOpContext BW_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"), TORCH_FN(conv2d_clamp_prepack));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_prepack"), TORCH_FN(linear_prepack));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"), TORCH_FN(conv2d_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("vulkan_prepack::linear_run"), TORCH_FN(linear_run));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
