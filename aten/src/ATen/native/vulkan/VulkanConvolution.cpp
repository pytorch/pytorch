#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/vulkan/VulkanAten.h>
#include <ATen/native/vulkan/VulkanCommon.h>
#include <ATen/native/vulkan/VulkanConvolution.h>

namespace at {
namespace native {
namespace vulkan {
namespace detail {
namespace convolution2d {

namespace {
// TODO: This function is not used.
bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  return at::native::is_vulkan_available() && (4 == weight.ndimension()) &&
      (at::Backend::CPU == weight.options().backend()) &&
      (kFloat == weight.scalar_type());
}

} // namespace

c10::intrusive_ptr<vulkan::Conv2dOpContext> createConv2dClampPrePackOpContext(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return vulkan::VulkanConv2dOpContext::create_context(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      output_min,
      output_max);
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<at::native::vulkan::Conv2dOpContext>& op_context) {
  return op_context->run(input);
}

ContextConv2D create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto dilation_expanded =
      expand_param_if_needed(dilation, "dilation", 2);
  const Tensor weight_nchw = weight.contiguous();
  const auto ws = weight_nchw.sizes();
  return ContextConv2D{
      groups == 1 ? at::native::vulkan::convolution_prepack_weights(weight_nchw)
                  : weight_nchw.vulkan(),
      bias.has_value() ? c10::make_optional((*bias).vulkan()) : c10::nullopt,
      // TODO: Are we sure these tensors will always come into this fucntion with the
      // the dimensions expected below? What if they don't?  This may trigger a segfault.
      // TODO: If we need TORCH_CHECK(available()) calls here as a sanity check, add it.
      {{ws[0], ws[1], ws[2], ws[3]}},
      {padding_expanded[0], padding_expanded[1]},
      {stride_expanded[0], stride_expanded[1]},
      {dilation_expanded[0], dilation_expanded[1]},
      groups,
      output_min,
      output_max};
}

Tensor run(const ContextConv2D& context, const Tensor& input) {
  return at::native::vulkan::convolution_prepacked(
      input,
      context.weight_size_,
      context.weight_prepacked_vulkan_,
      context.bias_vulkan_,
      context.padding_,
      context.stride_,
      context.dilation_,
      context.groups_,
      context.output_min_,
      context.output_max_);
}

} // namespace convolution2d
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
