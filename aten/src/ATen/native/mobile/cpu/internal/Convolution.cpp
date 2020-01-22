#include <ATen/native/mobile/cpu/Engine.h>

#ifdef USE_XNNPACK

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/mobile/cpu/internal/Allocator.h>
#include <ATen/native/mobile/cpu/internal/Common.h>
#include <ATen/native/mobile/internal/ThreadPool.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {
namespace convolution {

struct Context final {
  Operator convolution_op;

  struct Output final {
    struct Spatial final {
      int64_t alpha;
      int64_t beta;
    };

    int64_t channels;
    Spatial height;
    Spatial width;
  } output;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

namespace {

// Supports NHWC and NCHW FP32 Activation4D:::: with any valid
//  - kernel size
//  - padding
//  - stride
//  - dilation
//  - grouping

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const Scalar output_min,
    const Scalar output_max) {
         // Mobile
  return mobile::cpu::available() &&
         // Weight
         (4 == weight.ndimension()) &&
         (weight.size(Layout::Filter::Height) > 0) &&
         (weight.size(Layout::Filter::Width) > 0) &&
         (c10::DeviceType::CPU == weight.device().type()) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                      (c10::DeviceType::CPU == bias->device().type()) &&
                                      (kFloat == bias->scalar_type()) &&
                                      (weight.size(Layout::Filter::Output)) == bias->size(0))
                                    : true) &&
         // Padding
         (padding[Layout::Parameter::Height] >= 0) &&
         (padding[Layout::Parameter::Width] >= 0) &&
         // Stride
         (stride[Layout::Parameter::Height] > 0) &&
         (stride[Layout::Parameter::Width] > 0) &&
         // Dilation
         (dilation[Layout::Parameter::Height] > 0) &&
         (dilation[Layout::Parameter::Width] > 0) &&
         // Groups
         (groups > 0) &&
         // Transpose
         !transposed &&
         // Input
         (weight.size(Layout::Filter::Input) > 0) &&
         // Output
         (weight.size(Layout::Filter::Output) > 0) &&
         // Output - Groups
         ((weight.size(Layout::Filter::Output) % groups) == 0) &&
         // Output Min / Max
         (output_min.isIntegral(true) || output_min.isFloatingPoint()) &&
         (output_max.isIntegral(true) || output_max.isFloatingPoint()) &&
         (output_max.to<float>() > output_min.to<float>());
}

Context create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding_,
    const IntArrayRef stride_,
    const IntArrayRef dilation_,
    const int64_t groups,
    const bool transposed,
    const Scalar output_min,
    const Scalar output_max) {
  const auto padding = expand_param_if_needed(padding_, "padding", 2);
  const auto stride = expand_param_if_needed(stride_, "stride", 2);
  const auto dilation = expand_param_if_needed(dilation_, "dilation", 2);

  TORCH_INTERNAL_ASSERT(
      available(
          weight,
          bias,
          padding,
          stride,
          dilation,
          groups,
          transposed,
          output_min,
          output_max),
      "mobile::cpu::convolution not available!");

  xnn_operator_t convolution_op{};

  const xnn_status create_status = xnn_create_convolution2d_nhwc_f32(
      padding[Layout::Parameter::Height],                               // input_padding_top
      padding[Layout::Parameter::Width],                                // input_padding_right
      padding[Layout::Parameter::Height],                               // input_padding_bottom
      padding[Layout::Parameter::Width],                                // input_padding_left
      weight.size(Layout::Filter::Height),                              // kernel_height
      weight.size(Layout::Filter::Width),                               // kernel_width
      stride[Layout::Parameter::Height],                                // subsampling_height
      stride[Layout::Parameter::Width],                                 // subsampling_width
      dilation[Layout::Parameter::Height],                              // dilation_height
      dilation[Layout::Parameter::Width],                               // dilation_width
      groups,                                                           // groups
      weight.size(Layout::Filter::Input),                               // group_input_channels
      weight.size(Layout::Filter::Output) / groups,                     // group_output_channels
      weight.size(Layout::Filter::Input) * groups,                      // input_pixel_stride
      weight.size(Layout::Filter::Output),                              // output_pixel_stride
      weight.contiguous(MemoryFormat::ChannelsLast).data_ptr<float>(),  // kernel
      (bias && bias->defined()) ? bias->data_ptr<float>() : nullptr,    // bias
      output_min.to<float>(),                                           // output_min
      output_max.to<float>(),                                           // output_max
      0u,                                                               // flags
      &convolution_op);                                                 // operator

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == create_status,
      "xnn_create_convolution2d_nhwc_f32 failed!");

  const auto Spatial = [&](const size_t filter, const size_t parameter) {
    const int64_t kernel = 1 + dilation[parameter] * (weight.size(filter) - 1);

    return Context::Output::Spatial{
      stride[parameter] - kernel + 2 * padding[parameter],
      stride[parameter],
    };
  };

  return Context{
    Operator(convolution_op),
    {
      weight.size(Layout::Filter::Output),
      Spatial(Layout::Filter::Height, Layout::Parameter::Height),
      Spatial(Layout::Filter::Width, Layout::Parameter::Width),
    },
  };
}

bool usable(const Tensor& input) {
         // Input
  return (4 == input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         (input.size(Layout::Activation4D::Batch) > 0) &&
         (input.size(Layout::Activation4D::Channels) > 0) &&
         (input.size(Layout::Activation4D::Height) > 0) &&
         (input.size(Layout::Activation4D::Width) > 0);
}

Tensor run(
    const Context& context,
    const Tensor& input) {
  using namespace mobile::internal;

  TORCH_INTERNAL_ASSERT(
      usable(input),
      "mobile::cpu::convolution not usable!");

  const auto Spatial = [](const int64_t input, const Context::Output::Spatial spatial) {
    return (input + spatial.alpha) / spatial.beta;
  };

  Tensor output = new_tensor(
      std::array<int64_t, 4>{
        input.size(Layout::Activation4D::Batch),
        context.output.channels,
        Spatial(input.size(Layout::Activation4D::Height), context.output.height),
        Spatial(input.size(Layout::Activation4D::Width), context.output.width),
      },
      input.options(),
      MemoryFormat::ChannelsLast);

  const xnn_status setup_status = xnn_setup_convolution2d_nhwc_f32(
      context.convolution_op.get(),                                   // operator
      input.size(Layout::Activation4D::Batch),                        // batch_size
      input.size(Layout::Activation4D::Height),                       // input_height
      input.size(Layout::Activation4D::Width),                        // input_width
      input.contiguous(MemoryFormat::ChannelsLast).data_ptr<float>(), // input
      output.data_ptr<float>(),                                       // output
      threadpool().handle());                                         // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == setup_status,
      "xnn_setup_convolution2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.convolution_op.get(), // operator
      threadpool().handle());       // threadpool

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
    const bool transposed,
    const Scalar output_min,
    const Scalar output_max) {
  return run(
      create(
          weight,
          bias,
          padding,
          stride,
          dilation,
          groups,
          transposed,
          output_min,
          output_max),
      input);
}

const auto registry = c10::RegisterOperators()
  .op("mobile::conv2d_create",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const Tensor& weight,
          const c10::optional<Tensor>& bias,
          const c10::List<int64_t> padding,
          const c10::List<int64_t> stride,
          const c10::List<int64_t> dilation,
          const int64_t groups,
          const bool transposed,
          const Scalar output_min,
          const Scalar output_max) {
        return cpp_custom_type_hack::create(
            std::make_unique<Context>(
                create(
                    weight,
                    bias,
                    padding.vec(),
                    stride.vec(),
                    dilation.vec(),
                    groups,
                    transposed,
                    output_min,
                    output_max)),
            weight.options());
      }))
  .op("mobile::conv2d_run",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const Tensor& context,
          const Tensor& input) {
        return run(
            cpp_custom_type_hack::cast<Context>(context),
            input);
      }));

} // namespace
} // namespace convolution
} // namespace internal

bool use_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed) {
  return internal::convolution::available(
            weight,
            bias,
            padding,
            stride,
            dilation,
            groups,
            transposed,
            internal::convolution::Context::kMin,
            internal::convolution::Context::kMax) &&
         internal::convolution::usable(input);
}

Tensor convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed) {
  return internal::convolution::create_and_run(
      input,
      weight,
      bias,
      padding,
      stride,
      dilation,
      groups,
      transposed,
      internal::convolution::Context::kMin,
      internal::convolution::Context::kMax);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::mobile::cpu::internal::convolution::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
