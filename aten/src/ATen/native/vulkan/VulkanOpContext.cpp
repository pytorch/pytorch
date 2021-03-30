#include <ATen/native/vulkan/VulkanOpContext.h>
#include <ATen/native/vulkan/VulkanConvolution.h>

namespace at {
namespace native {
namespace vulkan {

c10::intrusive_ptr<Conv2dOpContext> VulkanConv2dOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  auto op_context = vulkan::detail::convolution2d::create(
      weight,
      bias,
      padding,
      stride,
      dilation,
      groups,
      output_min ? output_min->to<float>() : vulkan::ContextConv2D::kMin,
      output_max ? output_max->to<float>() : vulkan::ContextConv2D::kMax);
  return c10::make_intrusive<VulkanConv2dOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      output_min,
      output_max,
      std::move(op_context));
}

Tensor VulkanConv2dOpContext::run(const Tensor& input) {
  return vulkan::detail::convolution2d::run(op_context_, input);
}

} // namespace vulkan
} // namespace native
} // namespace at
