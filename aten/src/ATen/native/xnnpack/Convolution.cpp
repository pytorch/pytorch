#ifdef USE_XNNPACK

#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace convolution2d {

struct Context final {
  Operator convolution_op;

  std::vector<int64_t> weight_size;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

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

Context create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding_,
    const IntArrayRef stride_,
    const IntArrayRef dilation_,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  const auto padding = expand_param_if_needed(padding_, "padding", 2);
  const auto stride = expand_param_if_needed(stride_, "stride", 2);
  const auto dilation = expand_param_if_needed(dilation_, "dilation", 2);
  const Tensor weight_nhwc = weight.contiguous(MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      available(
          weight_nhwc,
          bias,
          padding,
          stride,
          dilation,
          groups,
          output_min,
          output_max),
      "XNNPACK Convolution not available! "
      "Reason: The provided (weight, bias, padding, stride, dilation, groups, output_min, output_max) "
      "parameters are either invalid individually or their combination is not supported by XNNPACK.");

  xnn_operator_t convolution_op{};

  const xnn_status create_status = xnn_create_convolution2d_nhwc_f32(
      padding[Layout::Parameter::height],                             // input_padding_top
      padding[Layout::Parameter::width],                              // input_padding_right
      padding[Layout::Parameter::height],                             // input_padding_bottom
      padding[Layout::Parameter::width],                              // input_padding_left
      weight_nhwc.size(Layout::Filter::height),                       // kernel_height
      weight_nhwc.size(Layout::Filter::width),                        // kernel_width
      stride[Layout::Parameter::height],                              // subsampling_height
      stride[Layout::Parameter::width],                               // subsampling_width
      dilation[Layout::Parameter::height],                            // dilation_height
      dilation[Layout::Parameter::width],                             // dilation_width
      groups,                                                         // groups
      weight_nhwc.size(Layout::Filter::input),                        // group_input_channels
      weight_nhwc.size(Layout::Filter::output) / groups,              // group_output_channels
      weight_nhwc.size(Layout::Filter::input) * groups,               // input_pixel_stride
      weight_nhwc.size(Layout::Filter::output),                       // output_pixel_stride
      weight_nhwc.data_ptr<float>(),                                  // kernel
      (bias && bias->defined()) ? bias->data_ptr<float>() : nullptr,  // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &convolution_op);                                               // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_convolution2d_nhwc_f32 failed!");

  return Context{
      Operator(convolution_op),
      weight_nhwc.sizes().vec(),
      padding,
      stride,
      dilation,
  };
}

// TODO: Decouple and improve error handling and messages.
bool usable(const Tensor& input) {
         // Input
  return (4 == input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         (input.size(Layout::Activation4D::batch) > 0) &&
         (input.size(Layout::Activation4D::channels) > 0) &&
         (input.size(Layout::Activation4D::height) > 0) &&
         (input.size(Layout::Activation4D::width) > 0) &&
         true;
}

Tensor run(
    const Context& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor input_nhwc = input.contiguous(MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      usable(input_nhwc),
      "XNNPACK Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  Tensor output = empty_with_tail_padding(
      conv_output_size(
          input_nhwc.sizes(),
          context.weight_size,
          context.padding,
          context.stride,
          context.dilation),
      input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast);

  const xnn_status setup_status = xnn_setup_convolution2d_nhwc_f32(
      context.convolution_op.get(),                   // operator
      input_nhwc.size(Layout::Activation4D::batch),   // batch_size
      input_nhwc.size(Layout::Activation4D::height),  // input_height
      input_nhwc.size(Layout::Activation4D::width),   // input_width
      input_nhwc.data_ptr<float>(),                   // input
      output.data_ptr<float>(),                       // output
      nullptr);                                       // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_convolution2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.convolution_op.get(), // operator
      nullptr);                     // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output.contiguous(input.suggest_memory_format());
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
            internal::convolution2d::Context::kMin,
            internal::convolution2d::Context::kMax) &&
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
      internal::convolution2d::Context::kMin,
      internal::convolution2d::Context::kMax);
}

} // namespace xnnpack

at::Tensor _conv2d_prepack(
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const c10::optional<double> output_min,
    const c10::optional<double> output_max) {
  return cpp_custom_type_hack::create(
      std::make_unique<xnnpack::internal::convolution2d::Context>(
          xnnpack::internal::convolution2d::create(
              weight,
              bias,
              padding.vec(),
              stride.vec(),
              dilation.vec(),
              groups,
              output_min ? *output_min : xnnpack::internal::convolution2d::Context::kMin,
              output_max ? *output_max : xnnpack::internal::convolution2d::Context::kMax)),
      weight.options());
}

at::Tensor _conv2d_packed(
    const Tensor& packed_weight,
    const Tensor& input) {
  return xnnpack::internal::convolution2d::run(
      cpp_custom_type_hack::cast<xnnpack::internal::convolution2d::Context>(packed_weight),
      input);
}

} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::xnnpack::internal::convolution2d::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
