#ifdef USE_XNNPACK

#include <vector>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/util/irange.h>

namespace at::native::xnnpack {
namespace internal::convolution2d {

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
    const at::OptionalIntArrayRef bias_sizes_opt,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::available() &&
         // Weight
         (4 == weight.ndimension()) &&
         (weight.size(Layout::Filter::height) > 0) &&
         (weight.size(Layout::Filter::width) > 0) &&
         (weight.device().is_cpu()) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         (bias_sizes_opt.has_value() ? ((1 == bias_sizes_opt->size()) &&
                ((transposed ? (weight.size(Layout::Filter::input) ==
                                ((*bias_sizes_opt)[0] / groups))
                  : (weight.size(Layout::Filter::output) == ((*bias_sizes_opt)[0])))))
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
         (input.device().is_cpu()) &&
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
    const IntArrayRef output_padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const float output_min,
    const float output_max) {
  auto op_context = create(
      weight,
      bias,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      transposed,
      output_min,
      output_max);
  return run(op_context, input);
}

// XNNPack's deconvolution operator expects weights to be indexed in the following order:
//   * Groups
//   * Group Output Channels
//   * Kernel Height
//   * Kernel Width
//   * Group Input Channels
//
// (ref: https://github.com/google/XNNPACK/blob/ecd8311c8fd3d9ab47edbc3df5f2b5de7dabe75f/test/deconvolution-operator-tester.h#L678)
//
// This function takes in a contiguous NHWC pytorch tensor (e.g. MemoryFormat == ChannelsLast) and rearranges the weights in preparation for use with xnnpack.
// By default, for pytorch, transpose conv2d weights are {input_channels, output_Channels_per_group, kernel_height, kernel_width}.
// In addition, it condenses the tensor from 5 to 4 dimensions as expected by the rest of the pytorch framework by combining the groups and input_channels dimension.
const Tensor reorder_weights_for_transpose_conv(const Tensor& weight_nhwc,
    int num_groups) {

  TORCH_CHECK(weight_nhwc.size(0) % num_groups == 0, "The number of groups cannot be satisfied by the provided weight tensor.");

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  int input_channels_per_group = weight_nhwc.size(0) / num_groups;
  int output_channels_per_group = weight_nhwc.size(1);
  int kernel_width = weight_nhwc.size(3);
  int kernel_height = weight_nhwc.size(2);

  int o_offset = 1;
  int h_offset = (output_channels_per_group);
  int w_offset = (output_channels_per_group)*(kernel_height);
  int i_offset = (output_channels_per_group)*(kernel_height)*(kernel_width);
  int g_offset = (output_channels_per_group)*(kernel_height)*(kernel_width)*(input_channels_per_group);

  Tensor reordered = mobile::empty_with_tail_padding(
     weight_nhwc.sizes(),
     weight_nhwc.options().dtype(),
     MemoryFormat::ChannelsLast,
     weight_nhwc.opt_names());

  float* out_ptr = reordered.data_ptr<float>();
  float* in_ptr = weight_nhwc.data_ptr<float>();

  int out_index = 0;
  for (const auto g : c10::irange(num_groups)) {
    for (const auto o : c10::irange(output_channels_per_group)) {
      for (const auto w : c10::irange(kernel_width)) {
        for (const auto h : c10::irange(kernel_height)) {
          for (const auto i : c10::irange(input_channels_per_group)) {
            int in_index = (g*g_offset) + (i*i_offset) + (h*h_offset) + (w*w_offset) + (o*o_offset);
            out_ptr[out_index] = in_ptr[in_index];
            out_index++;
          }
        }
      }
    }
  }

  return reordered;
}

} // namespace

ContextConv2D create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef output_padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const float output_min,
    const float output_max) {
  const auto padding_expanded = expand_param_if_needed(padding, "padding", 2);
  const auto output_padding_expanded = expand_param_if_needed(output_padding, "output_padding", 2);
  const auto stride_expanded = expand_param_if_needed(stride, "stride", 2);
  const auto dilation_expanded = expand_param_if_needed(dilation, "dilation", 2);
  const Tensor weight_nhwc = weight.contiguous(MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      available(
          weight_nhwc,
          (bias.has_value() && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : std::nullopt,
          padding_expanded,
          stride_expanded,
          dilation_expanded,
          groups,
          transposed,
          output_min,
          output_max),
      "xnnpack::convolution not available! "
      "Reason: The provided (weight, bias, padding, stride, dilation, groups, transposed, output_min, output_max) "
      "parameters are either invalid individually or their combination is not supported by XNNPACK.");


  xnn_operator_t convolution_op{};
  xnn_status create_status{};
  std::array<int64_t, 4> weight_sizes{};

  if (transposed) {
    const Tensor weight_reordered = reorder_weights_for_transpose_conv(weight_nhwc, groups);
    for (const auto i : c10::irange(4)) {
      weight_sizes[i] = weight_reordered.size(i);
    }
    create_status = xnn_create_deconvolution2d_nhwc_f32(
      padding_expanded[Layout::Parameter::height],                    // output_padding_top
      padding_expanded[Layout::Parameter::width],                     // output_padding_right
      padding_expanded[Layout::Parameter::height],                    // output_padding_bottom
      padding_expanded[Layout::Parameter::width],                     // output_padding_left
      weight_reordered.size(Layout::Filter::height),                  // kernel_height
      weight_reordered.size(Layout::Filter::width),                   // kernel_width
      stride_expanded[Layout::Parameter::height],                     // subsampling_height
      stride_expanded[Layout::Parameter::width],                      // subsampling_width
      dilation_expanded[Layout::Parameter::height],                   // dilation_height
      dilation_expanded[Layout::Parameter::width],                    // dilation_width
      groups,                                                         // groups
      weight_reordered.size(Layout::Filter::output) / groups,         // group_input_channels
      weight_reordered.size(Layout::Filter::input),                   // group_output_channels
      weight_reordered.size(Layout::Filter::output),                  // input_pixel_stride
      weight_reordered.size(Layout::Filter::input) * groups,          // output_pixel_stride
      weight_reordered.data_ptr<float>(),                             // kernel
      (bias && bias->defined())
          ? bias->contiguous().data_ptr<float>()
          : nullptr,                                                  // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      nullptr,                                                        // xnn_caches_t
      nullptr,                                                        // xnn_weights_cache_t
      &convolution_op);                                               // operator
  } else {
    for (const auto i : c10::irange(4)) {
      weight_sizes[i] = weight_nhwc.size(i);
    }
    create_status = xnn_create_convolution2d_nhwc_f32(
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
      nullptr,                                                        // xnn_caches_t
      nullptr,                                                        // xnn_weights_cache_t
      &convolution_op);                                               // operator
  }

  TORCH_CHECK(
      xnn_status_success == create_status,
      (transposed ? "xnn_create_deconvolution2d_nhwc_f32 failed!"
                  : "xnn_create_convolution2d_nhwc_f32 failed!"));

  return ContextConv2D{
      Operator(convolution_op),
      weight_sizes,
      {padding_expanded[0], padding_expanded[1]},
      {output_padding_expanded[0], output_padding_expanded[1]},
      {stride_expanded[0], stride_expanded[1]},
      {dilation_expanded[0], dilation_expanded[1]},
      transposed, groups
  };
}

Tensor run(
    ContextConv2D& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor padded_input_nhwc = mobile::allocate_padded_contiguous_if_needed(
      input, MemoryFormat::ChannelsLast);

  TORCH_CHECK(
      usable(padded_input_nhwc),
      "XNNPACK Convolution not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  Tensor output;
  if (context.transposed_) {
    output = mobile::empty_with_tail_padding(
      conv_input_size(padded_input_nhwc.sizes(),
        context.weight_size_,
        context.padding_,
        context.output_padding_,
        context.stride_,
        context.dilation_,
        context.groups_),
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.opt_names());
  } else {
    output = mobile::empty_with_tail_padding(
      conv_output_size(
          padded_input_nhwc.sizes(),
          context.weight_size_,
          context.padding_,
          context.stride_,
          context.dilation_),
      padded_input_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      padded_input_nhwc.opt_names());
  }

  xnn_status setup_status{};

  /*
   * Input Pointer Caching:
   * Previously, we cached the input/output pointers and dimension parameters
   * so that if the same pointers and parameters are used, this setup could be
   * skipped.
   * However, XNNPack has integrated offsets with its indirection buffer, so the
   * buffer does not need to be recalculated even if activation tensor pointer
   * changes as long as tensor dimensions are the same. Thus, the aforementioned
   * manual caching is not needed here.
   */

  if (context.transposed_) {
    setup_status = xnn_reshape_deconvolution2d_nhwc_f32(
      context.op.get(),
      padded_input_nhwc.size(Layout::Activation4D::batch),   // batch_size
      padded_input_nhwc.size(Layout::Activation4D::height),  // input_height
      padded_input_nhwc.size(Layout::Activation4D::width),   // input_width
      context.output_padding_[0],                            // adjustment_height
      context.output_padding_[1],                            // adjustment_width
      nullptr,                                               // output_height_out
      nullptr,                                               // output_width_out
      caffe2::pthreadpool_());                               // threadpool

    setup_status = xnn_setup_deconvolution2d_nhwc_f32(
      context.op.get(),                                      // operator
      padded_input_nhwc.data_ptr<float>(),                   // input
      output.data_ptr<float>());                             // output
  } else {
    size_t workspace_size = SIZE_MAX;
    size_t workspace_alignment = SIZE_MAX;

    setup_status = xnn_reshape_convolution2d_nhwc_f32(
      context.op.get(),
      padded_input_nhwc.size(Layout::Activation4D::batch),   // batch_size
      padded_input_nhwc.size(Layout::Activation4D::height),  // input_height
      padded_input_nhwc.size(Layout::Activation4D::width),   // input_width
      &workspace_size,                                       // workspace_size
      &workspace_alignment,                                  // workspace_alignment
      nullptr,                                               // output_height_out
      nullptr,                                               // output_width_out
      caffe2::pthreadpool_());

    setup_status = xnn_setup_convolution2d_nhwc_f32(
      context.op.get(),                                      // operator
      nullptr,                                               // workspace
      padded_input_nhwc.data_ptr<float>(),                   // input
      output.data_ptr<float>());                             // output
  }

  TORCH_CHECK(
      xnn_status_success == setup_status,
      (context.transposed_ ? "xnn_setup_deconvolution2d_nhwc_f32 failed!"
                            : "xnn_setup_convolution2d_nhwc_f32 failed!"));

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
        std::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const std::optional<Scalar>& output_min,
        const std::optional<Scalar>& output_max) {
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

c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>
    createConv2dTransposeClampPrePackOpContext(
        Tensor weight,
        std::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> output_padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const std::optional<Scalar>& output_min,
        const std::optional<Scalar>& output_max) {
      return xnnpack::XNNPackTransposeConv2dOpContext::create_context(
          std::move(weight),
          std::move(bias),
          std::move(padding),
          std::move(output_padding),
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

// Op is registered to have Any argument as we plan to reuse it for prepacked conv2d of other backends
IValue
unpack_prepacked_sizes_conv2d(const IValue& ivalue) {
  auto op_context = ivalue.toCustomClass<xnnpack::Conv2dOpContext>();
  const auto tuple = op_context->unpack();
  const auto& bias = std::get<1>(tuple);
  return IValue(std::make_tuple(
      std::get<0>(tuple).sizes(),
      (bias && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : std::nullopt,
      std::get<2>(tuple),
      std::get<3>(tuple),
      std::get<4>(tuple),
      std::get<5>(tuple)));
}

Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>& op_context) {
  return op_context->run(input);
}

} // namespace internal

bool use_convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const at::OptionalIntArrayRef bias_sizes_opt,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed) {
  return internal::convolution2d::available(
            weight,
            bias_sizes_opt,
            padding,
            stride,
            dilation,
            groups,
            transposed,
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
      {0, 0}, // output_padding
      stride,
      dilation,
      groups,
      false,  // transposed
      ContextConv2D::kMin,
      ContextConv2D::kMax);
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
