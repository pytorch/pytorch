#ifdef USE_XNNPACK
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at {
namespace native {
namespace xnnpack {

c10::intrusive_ptr<XNNPackLinearOpContext>
XNNPackLinearOpContext::create_context(at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    const c10::optional<double> output_min,
    const c10::optional<double> output_max) {
  auto linear_op_context =
      c10::make_intrusive<XNNPackLinearOpContext>(std::move(weight), std::move(bias));
  linear_op_context->op_context_ = xnnpack::internal::linear::create(
      linear_op_context->orig_weight_,
      linear_op_context->orig_bias_,
      output_min ? *output_min : xnnpack::ContextLinear::kMin,
      output_max ? *output_max : xnnpack::ContextLinear::kMax);
  return linear_op_context;
}

c10::intrusive_ptr<XNNPackConv2dOpContext>
XNNPackConv2dOpContext::create_context(at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const c10::optional<double> output_min,
    const c10::optional<double> output_max) {
  auto conv2d_op_context =
      c10::make_intrusive<XNNPackConv2dOpContext>(
          std::move(weight),
          std::move(bias),
          std::move(padding),
          std::move(stride),
          std::move(dilation),
          groups);
  conv2d_op_context->op_context_ =
      xnnpack::internal::convolution2d::create(
          conv2d_op_context->orig_weight_,
          conv2d_op_context->orig_bias_,
          conv2d_op_context->padding_,
          conv2d_op_context->stride_,
          conv2d_op_context->dilation_,
          conv2d_op_context->groups_,
          output_min ? *output_min : xnnpack::ContextConv2D::kMin,
          output_max ? *output_max : xnnpack::ContextConv2D::kMax);
  return conv2d_op_context;
}

} // xnnpack
} // native
} // at

#endif /* USE_XNNPACK */
