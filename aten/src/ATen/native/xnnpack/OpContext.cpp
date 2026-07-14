#ifdef USE_XNNPACK
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>

#include <ATen/Context.h>

namespace at::native::xnnpack {

c10::intrusive_ptr<LinearOpContext>
XNNPackLinearOpContext::create_context(
    at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  auto linear_op_context =
      c10::make_intrusive<XNNPackLinearOpContext>(
          std::move(weight),
          std::move(bias),
          output_min,
          output_max,
          xnnpack::internal::linear::create(
              weight,
              bias,
              output_min ? output_min->to<float>()
                         : xnnpack::ContextLinear::kMin,
              output_max ? output_max->to<float>()
                         : xnnpack::ContextLinear::kMax)
          );
  if (at::globalContext().releaseWeightsWhenPrepacking()) {
    linear_op_context->free_orig_weight_and_bias();
  }

  return linear_op_context;
}

void XNNPackLinearOpContext::free_orig_weight_and_bias() {
  orig_weight_and_bias_freed_ = true;
  orig_weight_.reset();
  orig_bias_.reset();
}

Tensor XNNPackLinearOpContext::run(const Tensor& input) {
  return xnnpack::internal::linear::run(op_context_, input);
}

c10::intrusive_ptr<Conv2dOpContext>
XNNPackConv2dOpContext::create_context(at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  auto op_context =
      xnnpack::internal::convolution2d::create(
          weight,
          bias,
          padding,
          {0, 0}, // output_padding
          stride,
          dilation,
          groups,
          false,  // transposed
          output_min ? output_min->to<float>()
                     : xnnpack::ContextConv2D::kMin,
          output_max ? output_max->to<float>()
                     : xnnpack::ContextConv2D::kMax);

  auto conv2d_op_context =
      c10::make_intrusive<XNNPackConv2dOpContext>(
          std::move(weight),
          std::move(bias),
          std::move(padding),
          std::move(stride),
          std::move(dilation),
          groups,
          output_min,
          output_max,
          std::move(op_context));

  if (at::globalContext().releaseWeightsWhenPrepacking()) {
    conv2d_op_context->free_orig_weight_and_bias();
  }

  return conv2d_op_context;
}

c10::intrusive_ptr<TransposeConv2dOpContext>
XNNPackTransposeConv2dOpContext::create_context(at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  auto op_context =
      xnnpack::internal::convolution2d::create(
          weight,
          bias,
          padding,
          output_padding,
          stride,
          dilation,
          groups,
          true, // transposed
          output_min ? output_min->to<float>()
                     : xnnpack::ContextConv2D::kMin,
          output_max ? output_max->to<float>()
                     : xnnpack::ContextConv2D::kMax);

  auto conv2d_op_context =
      c10::make_intrusive<XNNPackTransposeConv2dOpContext>(
          std::move(weight),
          std::move(bias),
          std::move(padding),
          std::move(output_padding),
          std::move(stride),
          std::move(dilation),
          groups,
          output_min,
          output_max,
          std::move(op_context));

  if (at::globalContext().releaseWeightsWhenPrepacking()) {
    conv2d_op_context->free_orig_weight_and_bias();
  }

  return conv2d_op_context;
}

Tensor XNNPackConv2dOpContext::run(const Tensor& input) {
  std::lock_guard<std::mutex> lock(xnnp_mutex_);
  return xnnpack::internal::convolution2d::run(op_context_, input);
}

Tensor XNNPackTransposeConv2dOpContext::run(const Tensor& input) {
  std::lock_guard<std::mutex> lock(xnnp_mutex_);
  return xnnpack::internal::convolution2d::run(op_context_, input);
}

void XNNPackConv2dOpContext::free_orig_weight_and_bias() {
  orig_weight_and_bias_freed_ = true;
  orig_weight_.reset();
  orig_bias_.reset();
}

void XNNPackTransposeConv2dOpContext::free_orig_weight_and_bias() {
  orig_weight_and_bias_freed_ = true;
  orig_weight_.reset();
  orig_bias_.reset();
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
