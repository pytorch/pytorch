#ifdef USE_XNNPACK

#include <ATen/native/Pool.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Pooling.h>

namespace at {
namespace native {
namespace xnnpack {

// Supports NHWC and NCHW FP32 max pooling with any
//  - kernel size
//  - padding
//  - stride
//  - dilation

bool use_max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  using namespace internal;

  // Make sure we are not dealing with an unorthodox configuration.
  if (kernel_.empty() || padding_.empty() || dilation_.empty()) {
    return false;
  }

  // Stride can be legitimately empty, in which case it is to be defaulted to kernel size.
  if (stride_.empty()) {
    stride_ = kernel_;
  }

  // Normalize the parameters.
  const internal::pooling::Parameters parameters{
    kernel_,
    padding_,
    stride_,
    dilation_,
  };

  // Here are the list of conditions required for this code path to be taken:
  // * Input must be 4D CPU float tensor with no gradients.
  // * Kernel must be a 2D IntArrayRef containing two positive numbers.
  //   Furthermore, 1x1 kernels are not valid as XNNPACK prohibits their use.
  // * Padding must be a 2D IntArrayRef containing two non-negative numbers.
  // * Stride must be a 2D IntArrayRef containing two positive numbers.
  // * Dilation must be a 2D IntArrayRef containing two positive numbers.
  // * Ceil mode is not supported and must be disabled.
  // * output_max must be greater than output_min.
  //   Namely, setting both output_min and output_max to 0 is not valid usage.
  // * Finally, application of this operator to the input tensor with the given
  //   max pool 2d parameters must result in an output tensor with a valid shape.
  const int64_t pt_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),
      parameters.kernel[Layout::Parameter::height],
      parameters.padding[Layout::Parameter::height],
      parameters.stride[Layout::Parameter::height],
      parameters.dilation[Layout::Parameter::height],
      ceil_mode);
  const int64_t pt_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),
      parameters.kernel[Layout::Parameter::width],
      parameters.padding[Layout::Parameter::width],
      parameters.stride[Layout::Parameter::width],
      parameters.dilation[Layout::Parameter::width],
      ceil_mode);
  const int64_t xnnpack_outputHeight = pooling_output_shape(
      input.size(Layout::Activation4D::height),
      parameters.kernel[Layout::Parameter::height],
      parameters.padding[Layout::Parameter::height],
      parameters.stride[Layout::Parameter::height],
      parameters.dilation[Layout::Parameter::height],
      false);
  const int64_t xnnpack_outputWidth = pooling_output_shape(
      input.size(Layout::Activation4D::width),
      parameters.kernel[Layout::Parameter::width],
      parameters.padding[Layout::Parameter::width],
      parameters.stride[Layout::Parameter::width],
      parameters.dilation[Layout::Parameter::width],
      false);

  const bool output_size_eq = (pt_outputHeight == xnnpack_outputHeight) &&
    (pt_outputWidth == xnnpack_outputWidth);

  return xnnpack::internal::available() &&
      // Input
      (4 == input.dim()) &&
      (input.device().is_cpu()) &&
      (kFloat == input.scalar_type()) &&
      !input.requires_grad() &&
      // Kernel
      (2 == parameters.kernel.size()) &&
      (parameters.kernel[Layout::Parameter::height] > 0) &&
      (parameters.kernel[Layout::Parameter::width] > 0) &&
      ((parameters.kernel[Layout::Parameter::height] *
        parameters.kernel[Layout::Parameter::width]) > 1) &&
      // Padding
      (2 == parameters.padding.size()) &&
      (parameters.padding[Layout::Parameter::height] >= 0) &&
      (parameters.padding[Layout::Parameter::width] >= 0) &&
      // Stride
      (2 == parameters.stride.size()) &&
      (parameters.stride[Layout::Parameter::height] > 0) &&
      (parameters.stride[Layout::Parameter::width] > 0) &&
      // Dilation
      (2 == parameters.dilation.size()) &&
      (parameters.dilation[Layout::Parameter::height] > 0) &&
      (parameters.dilation[Layout::Parameter::width] > 0) &&
      // Ceil Mode
      (!ceil_mode || output_size_eq) &&
      // Output Min / Max
      (output_max > output_min) &&
      // Output
      (pooling_output_shape(
        input.size(Layout::Activation4D::height),
        parameters.kernel[Layout::Parameter::height],
        parameters.padding[Layout::Parameter::height],
        parameters.stride[Layout::Parameter::height],
        parameters.dilation[Layout::Parameter::height],
        ceil_mode) > 0) &&
      (pooling_output_shape(
        input.size(Layout::Activation4D::width),
        parameters.kernel[Layout::Parameter::width],
        parameters.padding[Layout::Parameter::width],
        parameters.stride[Layout::Parameter::width],
        parameters.dilation[Layout::Parameter::width],
        ceil_mode) > 0) &&
      true;
}

Tensor max_pool2d(
    const Tensor& input,
    const IntArrayRef kernel_,
    const IntArrayRef padding_,
    IntArrayRef stride_,
    const IntArrayRef dilation_,
    const bool ceil_mode,
    const float output_min,
    const float output_max) {
  using namespace internal;

  // A call to max_pool2d must have been gated by a call to use_maxpool2d, so
  // the parameters are guaranteed to be valid at this point.  Still, stride can
  // be empty, and the parameters not normalized.

  if (stride_.empty()) {
    stride_ = kernel_;
  }

  const internal::pooling::Parameters parameters{
    kernel_,
    padding_,
    stride_,
    dilation_,
  };

  const Tensor input_padded_contig_nhwc =
      mobile::allocate_padded_contiguous_if_needed(
          input,
          MemoryFormat::ChannelsLast);

  Tensor output_padded_contig_nhwc = mobile::empty_with_tail_padding(
      {
        input_padded_contig_nhwc.size(Layout::Activation4D::batch),
        input_padded_contig_nhwc.size(Layout::Activation4D::channels),
        pooling_output_shape(
            input_padded_contig_nhwc.size(Layout::Activation4D::height),
            parameters.kernel[Layout::Parameter::height],
            parameters.padding[Layout::Parameter::height],
            parameters.stride[Layout::Parameter::height],
            parameters.dilation[Layout::Parameter::height],
            ceil_mode),
        pooling_output_shape(
            input_padded_contig_nhwc.size(Layout::Activation4D::width),
            parameters.kernel[Layout::Parameter::width],
            parameters.padding[Layout::Parameter::width],
            parameters.stride[Layout::Parameter::width],
            parameters.dilation[Layout::Parameter::width],
            ceil_mode),
      },
      input_padded_contig_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      input_padded_contig_nhwc.names());

  xnn_operator_t max_pool_op{};

  const xnn_status create_status = xnn_create_max_pooling2d_nhwc_f32(
      parameters.padding[Layout::Parameter::height],                  // input_padding_top
      parameters.padding[Layout::Parameter::width],                   // input_padding_right
      parameters.padding[Layout::Parameter::height],                  // input_padding_bottom
      parameters.padding[Layout::Parameter::width],                   // input_padding_left
      parameters.kernel[Layout::Parameter::height],                   // kernel_height
      parameters.kernel[Layout::Parameter::width],                    // kernel_width
      parameters.stride[Layout::Parameter::height],                   // subsampling_height
      parameters.stride[Layout::Parameter::width],                    // subsampling_width
      parameters.dilation[Layout::Parameter::height],                 // dilation_height
      parameters.dilation[Layout::Parameter::width],                  // dilation_width
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // channels
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // input_pixel_stride - NHWC Contiguous
      output_padded_contig_nhwc.size(Layout::Activation4D::channels), // output_pixel_stride - NHWC Contiguous
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &max_pool_op);                                                  // operator

  Operator max_pool_scoped_op(max_pool_op);

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_max_pooling2d_nhwc_f32 failed!");

  const xnn_status setup_status = xnn_setup_max_pooling2d_nhwc_f32(
      max_pool_op,                                                  // operator
      input_padded_contig_nhwc.size(Layout::Activation4D::batch),   // batch_size
      input_padded_contig_nhwc.size(Layout::Activation4D::height),  // input_height
      input_padded_contig_nhwc.size(Layout::Activation4D::width),   // input_width
      input_padded_contig_nhwc.data_ptr<float>(),                   // input
      output_padded_contig_nhwc.data_ptr<float>(),                  // output
      caffe2::pthreadpool_());                                      // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_max_pooling2d_nhwc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      max_pool_op,              // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output_padded_contig_nhwc.contiguous(input.suggest_memory_format());
}

} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
