#include <ATen/native/mobile/cpu/Engine.h>

#ifdef USE_XNNPACK

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/mobile/cpu/internal/Allocator.h>
#include <ATen/native/mobile/cpu/internal/Common.h>
#include <ATen/native/mobile/internal/ThreadPool.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
namespace internal {
namespace max_pool {

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
    const Scalar output_min,
    const Scalar output_max) {
         // Mobile
  return mobile::cpu::available() &&
         // Input
         (input_channels > 0) &&
         // Output
         (output_channels > 0) &&
         // Input == Output
         (input_channels == output_channels) &&
         // Kernel
         (kernel[Layout::Parameter::Height] > 0) &&
         (kernel[Layout::Parameter::Width] > 0) &&
         // Padding
         (padding[Layout::Parameter::Height] >= 0) &&
         (padding[Layout::Parameter::Width] >= 0) &&
         // Stride
         (stride[Layout::Parameter::Height] > 0) &&
         (stride[Layout::Parameter::Width] > 0) &&
         // Dilation
         (dilation[Layout::Parameter::Height] > 0) &&
         (dilation[Layout::Parameter::Width] > 0) &&
         // Output
         (output_min.isIntegral(true) || output_min.isFloatingPoint()) &&
         (output_max.isIntegral(true) || output_max.isFloatingPoint()) &&
         (output_max.to<float>() > output_min.to<float>());
}

Context create(
    const int64_t input_channels,
    const int64_t output_channels,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    const IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const Scalar output_min,
    const Scalar output_max) {
  const auto kernel = expand_param_if_needed(kernel_, "kernel", 2);
  const auto padding = expand_param_if_needed(padding_, "padding", 2);
  const auto stride = expand_param_if_needed(stride_, "stride", 2);
  const auto dilation = expand_param_if_needed(dilation_, "dilation", 2);

  TORCH_INTERNAL_ASSERT(
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
      padding[Layout::Parameter::Height],   // input_padding_top
      padding[Layout::Parameter::Width],    // input_padding_right
      padding[Layout::Parameter::Height],   // input_padding_bottom
      padding[Layout::Parameter::Width],    // input_padding_left
      kernel[Layout::Parameter::Height],    // kernel_height
      kernel[Layout::Parameter::Width],     // kernel_width
      stride[Layout::Parameter::Height],    // subsampling_height
      stride[Layout::Parameter::Width],     // subsampling_width
      dilation[Layout::Parameter::Height],  // dilation_height
      dilation[Layout::Parameter::Width],   // dilation_width
      input_channels,                       // channels,
      input_channels,                       // input_pixel_stride
      output_channels,                      // output_pixel_stride
      output_min.to<float>(),               // output_min
      output_max.to<float>(),               // output_max
      0u,                                   // flags
      &max_pool_op);                        // operator

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == create_status,
      "xnn_create_max_pooling2d_nhwc_f32 failed!");

  return Context{
    Operator(max_pool_op),
    {
      output_channels,
      {
        kernel[Layout::Parameter::Height],
        padding[Layout::Parameter::Height],
        stride[Layout::Parameter::Height],
        dilation[Layout::Parameter::Height],
        ceil_mode,
      },
      {
        kernel[Layout::Parameter::Width],
        padding[Layout::Parameter::Width],
        stride[Layout::Parameter::Width],
        dilation[Layout::Parameter::Width],
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
  using namespace mobile::internal;

  TORCH_INTERNAL_ASSERT(
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

  Tensor output = new_tensor(
      {
        input.size(Layout::Activation4D::Batch),
        context.output.channels,
        Spatial(input.size(Layout::Activation4D::Height), context.output.height),
        Spatial(input.size(Layout::Activation4D::Width), context.output.width),
      },
      input.options(),
      MemoryFormat::ChannelsLast);

  const xnn_status setup_status = xnn_setup_max_pooling2d_nhwc_f32(
      context.max_pool_op.get(),                                      // operator
      input.size(Layout::Activation4D::Batch),                        // batch_size
      input.size(Layout::Activation4D::Height),                       // input_height
      input.size(Layout::Activation4D::Width),                        // input_width
      input.contiguous(MemoryFormat::ChannelsLast).data_ptr<float>(), // input
      output.data_ptr<float>(),                                       // output
      threadpool().handle());                                         // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == setup_status,
      "xnn_setup_max_pooling2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.max_pool_op.get(),  // operator
      threadpool().handle());     // threadpool

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
    const Scalar output_min,
    const Scalar output_max) {
  using namespace mobile::internal;

  const int64_t input_channels = input.size(Layout::Activation4D::Channels);
  const int64_t output_channels = input_channels;

  return internal::max_pool::run(
      internal::max_pool::create(
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

const auto registry = c10::RegisterOperators()
  .op("mobile::max_pool2d_create",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const int64_t input_channels,
          const int64_t output_channels,
          const c10::List<int64_t> kernel,
          const c10::List<int64_t> padding,
          const c10::List<int64_t> stride,
          const c10::List<int64_t> dilation,
          const bool ceil_mode,
          const Scalar output_min,
          const Scalar output_max) {
        return cpp_custom_type_hack::create(
            std::make_unique<Context>(
                create(
                    input_channels,
                    output_channels,
                    kernel.vec(),
                    padding.vec(),
                    stride.vec(),
                    dilation.vec(),
                    ceil_mode,
                    output_min,
                    output_max)),
                c10::TensorOptions{});
      }))
  .op("mobile::max_pool2d_run",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const c10::optional<Tensor>& context,
          const Tensor& input) {
        TORCH_INTERNAL_ASSERT(context, "Invalid context!");

        return run(
            cpp_custom_type_hack::cast<Context>(*context),
            input);
      }))
  .op("mobile::max_pool2d",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const Tensor& input,
          const c10::List<int64_t> kernel,
          const c10::List<int64_t> padding,
          const c10::List<int64_t> stride,
          const c10::List<int64_t> dilation,
          const bool ceil_mode,
          const Scalar output_min,
          const Scalar output_max) {
        return create_and_run(
            input,
            kernel.vec(),
            padding.vec(),
            stride.vec(),
            dilation.vec(),
            ceil_mode,
            output_min,
            output_max);
      }));

} // namespace
} // namespace max_pool
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
                                 input.size(Layout::Activation4D::Channels) :
                                 -1;

  const int64_t output_channels = input_channels;

  return internal::max_pool::available(
            input_channels,
            output_channels,
            kernel,
            padding,
            stride,
            dilation,
            ceil_mode,
            internal::max_pool::Context::kMin,
            internal::max_pool::Context::kMax) &&
         internal::max_pool::usable(input);
}

Tensor max_pool(
    const Tensor& input,
    const IntArrayRef kernel,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const bool ceil_mode) {
  return internal::max_pool::create_and_run(
      input,
      kernel,
      padding,
      stride,
      dilation,
      ceil_mode,
      internal::max_pool::Context::kMin,
      internal::max_pool::Context::kMax);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::mobile::cpu::internal::max_pool::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
