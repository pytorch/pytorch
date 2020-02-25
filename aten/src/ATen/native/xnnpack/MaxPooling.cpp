#ifdef USE_XNNPACK

#include <ATen/native/Pool.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>
#include <ATen/native/xnnpack/Pooling.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace max_pool2d {

struct Context final {
  typedef pool::Output Output;

  Operator max_pool_op;
  Output output;

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
    const int64_t channels,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::internal::available() &&
         // Input / Output Channels
         (channels > 0) &&
         // Kernel
         (2 == kernel.size()) &&
         (kernel[Layout::Parameter::height] > 0) &&
         (kernel[Layout::Parameter::width] > 0) &&
         ((kernel[Layout::Parameter::height] * kernel[Layout::Parameter::width]) > 1) &&
         // Padding
         (2 == padding.size()) &&
         (padding[Layout::Parameter::height] >= 0) &&
         (padding[Layout::Parameter::width] >= 0) &&
         // Stride
         (2 == stride.size()) &&
         (stride[Layout::Parameter::height] > 0) &&
         (stride[Layout::Parameter::width] > 0) &&
         // Dilation
         (2 == dilation.size()) &&
         (dilation[Layout::Parameter::height] > 0) &&
         (dilation[Layout::Parameter::width] > 0) &&
         // Ceil Mode
         !ceil_mode &&
         // Output Min / Max
         (output_max > output_min) &&
         true;
}

Context create(
    const int64_t channels,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  TORCH_CHECK(
      available(
          channels,
          kernel,
          padding,
          stride,
          dilation,
          ceil_mode,
          output_min,
          output_max),
      "XNNPACK MaxPool2d not available! "
      "Reason: The provided (channels, kernel, padding, stride, dilation, ceil_mode, output_min, output_max) "
      "parameters are either invalid individually or their combination is not supported by XNNPACK.");

  xnn_operator_t max_pool_op{};

  const xnn_status create_status = xnn_create_max_pooling2d_nhwc_f32(
      padding[Layout::Parameter::height],   // input_padding_top
      padding[Layout::Parameter::width],    // input_padding_right
      padding[Layout::Parameter::height],   // input_padding_bottom
      padding[Layout::Parameter::width],    // input_padding_left
      kernel[Layout::Parameter::height],    // kernel_height
      kernel[Layout::Parameter::width],     // kernel_width
      stride[Layout::Parameter::height],    // subsampling_height
      stride[Layout::Parameter::width],     // subsampling_width
      dilation[Layout::Parameter::height],  // dilation_height
      dilation[Layout::Parameter::width],   // dilation_width
      channels,                             // channels
      channels,                             // input_pixel_stride - Always converted to NHWC contiguous prior to use.
      channels,                             // output_pixel_stride - Always in NHWC contiguous prior to use.
      output_min,                           // output_min
      output_max,                           // output_max
      0u,                                   // flags
      &max_pool_op);                        // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_max_pooling2d_nhwc_f32 failed!");

  return Context{
    Operator(max_pool_op),
    {
      channels,
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

int64_t shape(const int64_t input, const Context::Output::Spatial& spatial) {
  return pooling_output_shape(
      input,
      spatial.kernel,
      spatial.padding,
      spatial.stride,
      spatial.dilation,
      spatial.ceil_mode);
};

bool usable(
    const Tensor& input,
    const Context::Output::Spatial& height,
    const Context::Output::Spatial& width) {
         // Input
  return (4 == input.dim()) &&
         // (c10::DeviceType::CPU == input.device().type()) &&
         (at::Backend::CPU == input.options().backend()) &&
         (kFloat == input.scalar_type()) &&
         // Output
         (shape(input.size(Layout::Activation4D::height), height) > 0) &&
         (shape(input.size(Layout::Activation4D::width), width) > 0) &&
         true;
}

Tensor run(
    const Context& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor input_nhwc = input.contiguous(MemoryFormat::ChannelsLast);
  const Tensor padded_input_nhwc = allocate_padded_if_needed(input_nhwc);

  TORCH_CHECK(
      usable(padded_input_nhwc, context.output.height, context.output.width),
      "XNNPACK MaxPool2d not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  Tensor output_nhwc = empty_with_tail_padding(
      {
        padded_input_nhwc.size(Layout::Activation4D::batch),
        context.output.channels,
        shape(padded_input_nhwc.size(Layout::Activation4D::height), context.output.height),
        shape(padded_input_nhwc.size(Layout::Activation4D::width), context.output.width),
      },
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.names());

  const xnn_status setup_status = xnn_setup_max_pooling2d_nhwc_f32(
      context.max_pool_op.get(),                            // operator
      padded_input_nhwc.size(Layout::Activation4D::batch),  // batch_size
      padded_input_nhwc.size(Layout::Activation4D::height), // input_height
      padded_input_nhwc.size(Layout::Activation4D::width),  // input_width
      padded_input_nhwc.data_ptr<float>(),                  // input
      output_nhwc.data_ptr<float>(),                        // output
      caffe2::xnnpack_threadpool());                        // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_max_pooling2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.max_pool_op.get(),      // operator
      caffe2::xnnpack_threadpool());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output_nhwc.contiguous(input.suggest_memory_format());
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

  return internal::max_pool2d::run(
      internal::max_pool2d::create(
          input.size(Layout::Activation4D::channels),
          kernel,
          padding,
          stride,
          dilation,
          ceil_mode,
          output_min,
          output_max),
      input);
}

struct Parameters final {
  std::array<int64_t, 2> kernel;
  std::array<int64_t, 2> padding;
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> dilation;

  explicit Parameters(
      const IntArrayRef kernel_,
      const IntArrayRef padding_,
      const IntArrayRef stride_,
      const IntArrayRef dilation_)
  : kernel(normalize(kernel_)),
    padding(normalize(padding_)),
    stride(normalize(stride_)),
    dilation(normalize(dilation_)) {
  }

private:
  static std::array<int64_t, 2> normalize(const IntArrayRef parameter) {
    TORCH_INTERNAL_ASSERT(
        !parameter.empty(),
        "Invalid usage!  Reason: normalize() was called on an empty parameter.");

    return std::array<int64_t, 2>{
      parameter[0],
      (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
  }
};

} // namespace
} // namespace max_pool2d
} // namespace internal

bool use_max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode) {
  using namespace internal;

  if (input.dim() != 4) {
    return false;
  }

  // Make sure we are not dealing with an unorthodox configuration.
  if (kernel.empty() || padding.empty() || dilation.empty()) {
    return false;
  }

  // Stride can be legitimately empty, in which case it is to be defaulted to kernel size.
  if (stride.empty()) {
    stride = kernel;
  }

  // Normalize the parameters.
  const max_pool2d::Parameters parameters{
    kernel,
    padding,
    stride,
    dilation,
  };

  return max_pool2d::available(
        input.size(Layout::Activation4D::channels),
        parameters.kernel,
        parameters.padding,
        parameters.stride,
        parameters.dilation,
        ceil_mode,
        internal::max_pool2d::Context::kMin,
        internal::max_pool2d::Context::kMax) &&
     max_pool2d::usable(
        input,
        {
          parameters.kernel[Layout::Parameter::height],
          parameters.padding[Layout::Parameter::height],
          parameters.stride[Layout::Parameter::height],
          parameters.dilation[Layout::Parameter::height],
          ceil_mode,
        },
        {
          parameters.kernel[Layout::Parameter::width],
          parameters.padding[Layout::Parameter::width],
          parameters.stride[Layout::Parameter::width],
          parameters.dilation[Layout::Parameter::width],
          ceil_mode,
        });
}

Tensor max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode) {
  using namespace internal;

  // A call to max_pool2d must have been gated by a call to use_maxpool2d, so
  // the parameters are guaranteed to be valid at this point.  Still, stride can
  // be empty, and the parameters not normalized.

  if (stride.empty()) {
    stride = kernel;
  }

  const max_pool2d::Parameters parameters{
    kernel,
    padding,
    stride,
    dilation,
  };

  return max_pool2d::create_and_run(
      input,
      parameters.kernel,
      parameters.padding,
      parameters.stride,
      parameters.dilation,
      ceil_mode,
      internal::max_pool2d::Context::kMin,
      internal::max_pool2d::Context::kMax);
}

} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
