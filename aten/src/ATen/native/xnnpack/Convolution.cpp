#ifdef USE_XNNPACK

#include <vector>

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xnnpack/Factory.h>
#include <ATen/native/xnnpack/Convolution.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace convolution2d {

namespace {

// Supports NHWC and NCHW FP32 convolutions with any valid
//  - kernel size
//  - padding
//  - stride
//  - dilation
//  - grouping

// TODO: Decouple and improve error handling and messages.
bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::internal::available() &&
         // Weight
         (4 == weight.ndimension()) &&
         (weight.size(Layout::Filter::height) > 0) &&
         (weight.size(Layout::Filter::width) > 0) &&
         (c10::DeviceType::CPU == weight.device().type()) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                      (c10::DeviceType::CPU == bias->device().type()) &&
                                      (kFloat == bias->scalar_type()) &&
                                      (weight.size(Layout::Filter::output)) == bias->size(0))
                                    : true) &&
         // Padding
         (padding[Layout::Parameter::height] >= 0) &&
         (padding[Layout::Parameter::width] >= 0) &&
         // Stride
         (stride[Layout::Parameter::height] > 0) &&
         (stride[Layout::Parameter::width] > 0) &&
         // Dilation
         (dilation[Layout::Parameter::height] > 0) &&
         (dilation[Layout::Parameter::width] > 0) &&
         // Groups
         (groups > 0) &&
         // Input
         (weight.size(Layout::Filter::input) > 0) &&
         // Output
         (weight.size(Layout::Filter::output) > 0) &&
         // Output - Groups
         ((weight.size(Layout::Filter::output) % groups) == 0) &&
         // Output Min / Max
         (output_max > output_min) &&
         true;
}

// TODO: Decouple and improve error handling and messages.
bool usable(const Tensor& input) {
       // Input
  return (4 == input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         (input.size(Layout::Activation4D::batch) >= 0) &&
         (input.size(Layout::Activation4D::channels) > 0) &&
         (input.size(Layout::Activation4D::height) > 0) &&
         (input.size(Layout::Activation4D::width) > 0) &&
         !input.requires_grad() &&
         true;
}

Tensor create_and_run(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  return run(
      create(
          weight,
          bias,
          padding,
          stride,
          dilation,
          groups,
          output_min,
          output_max),
      input);
}

} // namespace

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
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);
  const Tensor weight_nhwc = weight.contiguous(MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      available(
          weight_nhwc,
          bias,
          padding_expanded,
          stride_expanded,
          dilation_expanded,
          groups,
          output_min,
          output_max),
      "xnnpack::convolution not available! "
      "Reason: The provided (weight, bias, padding, stride, dilation, groups, output_min, output_max) "
      "parameters are either invalid individually or their combination is not supported by XNNPACK.");

  xnn_operator_t convolution_op{};

  const xnn_status create_status = xnn_create_convolution2d_nhwc_f32(
      padding_expanded[Layout::Parameter::height],                    // input_padding_top
      padding_expanded[Layout::Parameter::width],                     // input_padding_right
      padding_expanded[Layout::Parameter::height],                    // input_padding_bottom
      padding_expanded[Layout::Parameter::width],                     // input_padding_left
      weight_nhwc.size(Layout::Filter::height),                       // kernel_height
      weight_nhwc.size(Layout::Filter::width),                        // kernel_width
      stride_expanded[Layout::Parameter::height],                     // subsampling_height
      stride_expanded[Layout::Parameter::width],                      // subsampling_width
      dilation_expanded[Layout::Parameter::height],                   // dilation_height
      dilation_expanded[Layout::Parameter::width],                    // dilation_width
      groups,                                                         // groups
      weight_nhwc.size(Layout::Filter::input),                        // group_input_channels
      weight_nhwc.size(Layout::Filter::output) / groups,              // group_output_channels
      weight_nhwc.size(Layout::Filter::input) * groups,               // input_pixel_stride
      weight_nhwc.size(Layout::Filter::output),                       // output_pixel_stride
      weight_nhwc.data_ptr<float>(),                                  // kernel
      (bias && bias->defined())
          ? bias->contiguous().data_ptr<float>()
          : nullptr,                                                  // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &convolution_op);                                               // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_convolution2d_nhwc_f32 failed!");

  return ContextConv2D{
      Operator(convolution_op),
      {weight_nhwc.sizes()[0], weight_nhwc.sizes()[1],
          weight_nhwc.sizes()[2], weight_nhwc.sizes()[3]},
      {padding_expanded[0], padding_expanded[1]},
      {stride_expanded[0], stride_expanded[1]},
      {dilation_expanded[0], dilation_expanded[1]}
  };
}

Tensor run(
    const ContextConv2D& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor padded_input_nhwc = allocate_padded_contiguous_if_needed(
      input, MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      usable(padded_input_nhwc),
      "XNNPACK Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  Tensor output = empty_with_tail_padding(
      conv_output_size(
          padded_input_nhwc.sizes(),
          context.weight_size_,
          context.padding_,
          context.stride_,
          context.dilation_),
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.names());

  const xnn_status setup_status = xnn_setup_convolution2d_nhwc_f32(
      context.op.get(),                                      // operator
      padded_input_nhwc.size(Layout::Activation4D::batch),   // batch_size
      padded_input_nhwc.size(Layout::Activation4D::height),  // input_height
      padded_input_nhwc.size(Layout::Activation4D::width),   // input_width
      padded_input_nhwc.data_ptr<float>(),                   // input
      output.data_ptr<float>(),                              // output
      caffe2::pthreadpool_());                               // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_convolution2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.op.get(),         // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output.contiguous(input.suggest_memory_format());
}

c10::intrusive_ptr<xnnpack::Conv2dOpContext>
    createConv2dClampPrePackOpContext(
        Tensor weight,
        c10::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        c10::optional<Scalar> output_min,
        c10::optional<Scalar> output_max) {
      return xnnpack::XNNPackConv2dOpContext::create_context(
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
    const c10::intrusive_ptr<xnnpack::Conv2dOpContext>& op_context) {
  return op_context->run(input);
}

} // namespace convolution2d
} // namespace internal

bool use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups) {
  return internal::convolution2d::available(
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
            ContextConv2D::kMin,
            ContextConv2D::kMax) &&
         internal::convolution2d::usable(input);
}

Tensor convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups) {
  return internal::convolution2d::create_and_run(
      input,
      weight,
      bias,
      padding,
      stride,
      dilation,
      groups,
      ContextConv2D::kMin,
      ContextConv2D::kMax);
}

} // namespace xnnpack

} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
