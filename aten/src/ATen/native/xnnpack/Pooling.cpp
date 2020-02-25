#ifdef USE_XNNPACK

#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace max_pool2d {

struct Context final {
  Operator max_pool_op;

  struct Output final {
    struct Spatial final {
      int64_t kernel;
      int64_t padding;
      int64_t stride;
      int64_t dilation;
      bool ceil_mode;
    };

    int64_t channels;
    Spatial height, width;
  } output;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

namespace {

// Supports NHWC and NCHW FP32 max pooling with any
//  - kernel size
//  - padding
//  - stride
//  - dilation

bool available(
    const int64_t input_channels,
    const int64_t output_channels,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool /* ceil_mode */,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::internal::available() &&
         // Input == Output
         (input_channels == output_channels) &&
         // Kernel
         (kernel[Layout::Parameter::height] > 0) &&
         (kernel[Layout::Parameter::width] > 0) &&
         // Padding
         (padding[Layout::Parameter::height] >= 0) &&
         (padding[Layout::Parameter::width] >= 0) &&
         // Stride
         (stride[Layout::Parameter::height] > 0) &&
         (stride[Layout::Parameter::width] > 0) &&
         // Dilation
         (dilation[Layout::Parameter::height] > 0) &&
         (dilation[Layout::Parameter::width] > 0) &&
         // Output Min / Max
         (output_max > output_min);
}

Context create(
    const int64_t input_channels,
    const int64_t output_channels,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    const IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  const auto kernel = expand_param_if_needed(kernel_, "kernel", 2);
  const auto padding = expand_param_if_needed(padding_, "padding", 2);
  const auto stride = expand_param_if_needed(stride_, "stride", 2);
  const auto dilation = expand_param_if_needed(dilation_, "dilation", 2);

  TORCH_CHECK(
      available(
          input_channels,
          output_channels,
          kernel,
          padding,
          stride,
          dilation,
          ceil_mode,
          output_min,
          output_max),
      "mobile::cpu::max_pool not available!");

  xnn_operator_t max_pool_op{};

  const xnn_status create_status = xnn_create_max_pooling2d_nhwc_f32(
      padding[Layout::Parameter::height],        // input_padding_top
      padding[Layout::Parameter::width],         // input_padding_right
      padding[Layout::Parameter::height],        // input_padding_bottom
      padding[Layout::Parameter::width],         // input_padding_left
      kernel[Layout::Parameter::height],         // kernel_height
      kernel[Layout::Parameter::width],          // kernel_width
      stride[Layout::Parameter::height],         // subsampling_height
      stride[Layout::Parameter::width],          // subsampling_width
      dilation[Layout::Parameter::height],       // dilation_height
      dilation[Layout::Parameter::width],        // dilation_width
      input_channels,                            // channels,
      input_channels,                            // input_pixel_stride
      output_channels,                           // output_pixel_stride
      output_min,                                // output_min
      output_max,                                // output_max
      0u,                                        // flags
      &max_pool_op);                             // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_max_pooling2d_nhwc_f32 failed!");

  return Context{
    Operator(max_pool_op),
    {
      output_channels,
      {
        kernel[Layout::Parameter::height],
        padding[Layout::Parameter::height],
        stride[Layout::Parameter::height],
        dilation[Layout::Parameter::height],
        ceil_mode,
      },
      {
        kernel[Layout::Parameter::width],
        padding[Layout::Parameter::width],
        stride[Layout::Parameter::width],
        dilation[Layout::Parameter::width],
        ceil_mode,
      },
    }
  };
}

bool usable(const Tensor& input) {
         // Input
  return (4u == input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type());
}

Tensor run(
    const Context& context,
    const Tensor& input) {
  using namespace internal;

  TORCH_CHECK(
      usable(input),
      "mobile::cpu::max_pool not usable!");

  const auto Spatial = [](const int64_t input, const Context::Output::Spatial& spatial) {
    return pooling_output_shape(
        input,
        spatial.kernel,
        spatial.padding,
        spatial.stride,
        spatial.dilation,
        spatial.ceil_mode);
  };

  Tensor output = empty_with_tail_padding(
      {
        input.size(Layout::Activation4D::batch),
        context.output.channels,
        Spatial(input.size(Layout::Activation4D::height), context.output.height),
        Spatial(input.size(Layout::Activation4D::width), context.output.width),
      },
      input.options().dtype(),
      MemoryFormat::ChannelsLast);

  const xnn_status setup_status = xnn_setup_max_pooling2d_nhwc_f32(
      context.max_pool_op.get(),                                      // operator
      input.size(Layout::Activation4D::batch),                        // batch_size
      input.size(Layout::Activation4D::height),                       // input_height
      input.size(Layout::Activation4D::width),                        // input_width
      input.contiguous(MemoryFormat::ChannelsLast).data_ptr<float>(), // input
      output.data_ptr<float>(),                                       // output
      nullptr);                                                       // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_max_pooling2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.max_pool_op.get(),  // operator
      nullptr);                   // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output.contiguous(input.suggest_memory_format());
}

Tensor create_and_run(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  using namespace internal;

  const int64_t input_channels = input.size(Layout::Activation4D::channels);
  const int64_t output_channels = input_channels;

  return internal::max_pool2d::run(
      internal::max_pool2d::create(
          input_channels,
          output_channels,
          kernel,
          padding,
          stride,
          dilation,
          ceil_mode,
          output_min,
          output_max),
      input);
}

} // namespace
} // namespace max_pool2d
} // namespace internal

bool use_max_pool(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode) {
  using namespace internal;

  const int64_t input_channels = (input.ndimension() == 4) ?
                                 input.size(Layout::Activation4D::channels) :
                                 -1;

  const int64_t output_channels = input_channels;

  return internal::max_pool2d::available(
            input_channels,
            output_channels,
            kernel,
            padding,
            stride,
            dilation,
            ceil_mode,
            internal::max_pool2d::Context::kMin,
            internal::max_pool2d::Context::kMax) &&
         internal::max_pool2d::usable(input);
}

Tensor max_pool(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode) {
  return internal::max_pool2d::create_and_run(
      input,
      kernel,
      padding,
      stride,
      dilation,
      ceil_mode,
      internal::max_pool2d::Context::kMin,
      internal::max_pool2d::Context::kMax);
}

} // namespace native
} // namespace at
} // namespace caffe2

#endif /* USE_XNNPACK */
