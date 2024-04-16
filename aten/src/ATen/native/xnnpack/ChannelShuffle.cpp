#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

namespace at::native::xnnpack {

bool use_channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
  using namespace internal;

  // Here are the list of conditions required for this code path to be taken:
  // * Input must be 4D CPU float tensor with no gradients and
  //   and all dimensions must be positive.
  // * The number of groups must be larger than 1 and
  //   the number of channels must be divisible by the number of groups.
  return xnnpack::available() &&
      // Input
      (4 == input.dim()) &&
      (input.device().is_cpu()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::Activation4D::batch) >= 0) &&
      (input.size(Layout::Activation4D::channels) > 0) &&
      (input.size(Layout::Activation4D::height) > 0) &&
      (input.size(Layout::Activation4D::width) > 0) &&
      !input.requires_grad() &&
      // Groups
      groups > 1 &&
      (0 == input.size(Layout::Activation4D::channels) % groups) &&
      true;
}

Tensor channel_shuffle(
    const Tensor& input,
    const int64_t groups) {
  using namespace internal;

  // A call to channel_shuffle must have been gated by a call to use_channel_shuffle,
  // so the parameters are guaranteed to be valid at this point.

  const Tensor input_padded_contig_nhwc =
      mobile::allocate_padded_contiguous_if_needed(
          input,
          MemoryFormat::ChannelsLast);

  Tensor output_padded_contig_nhwc = mobile::empty_with_tail_padding(
      {
        input_padded_contig_nhwc.size(Layout::Activation4D::batch),
        input_padded_contig_nhwc.size(Layout::Activation4D::channels),
        input_padded_contig_nhwc.size(Layout::Activation4D::height),
        input_padded_contig_nhwc.size(Layout::Activation4D::width),
      },
      input_padded_contig_nhwc.options().dtype(),
      MemoryFormat::ChannelsLast,
      input_padded_contig_nhwc.opt_names());

  int64_t channels_per_group =
      input_padded_contig_nhwc.size(Layout::Activation4D::channels) / groups;

  xnn_operator_t channel_shuffle_op{};

  const xnn_status create_status = xnn_create_channel_shuffle_nc_x32(
      groups,                                                         // number of groups
      channels_per_group,                                             // number of channels per group
      input_padded_contig_nhwc.size(Layout::Activation4D::channels),  // input_pixel_stride - NHWC Contiguous
      output_padded_contig_nhwc.size(Layout::Activation4D::channels), // output_pixel_stride - NHWC Contiguous
      0u,                                                             // flags
      &channel_shuffle_op);                                           // operator

  Operator channel_shuffle_scoped_op(channel_shuffle_op);

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_channel_shuffle_nc_x32 failed!");

  int64_t batch_size = input_padded_contig_nhwc.size(Layout::Activation4D::batch) *
                       input_padded_contig_nhwc.size(Layout::Activation4D::height) *
                       input_padded_contig_nhwc.size(Layout::Activation4D::width);

  const xnn_status reshape_status = xnn_reshape_channel_shuffle_nc_x32(
      channel_shuffle_op,                                           // operator
      batch_size,                                                   // batch_size
      caffe2::pthreadpool_());                                      // threadpool

  TORCH_CHECK(
      xnn_status_success == reshape_status,
      "xnn_reshape_channel_shuffle_nc_x32 failed!");

  const xnn_status setup_status = xnn_setup_channel_shuffle_nc_x32(
      channel_shuffle_op,                                           // operator
      input_padded_contig_nhwc.data_ptr<float>(),                   // input
      output_padded_contig_nhwc.data_ptr<float>());                 // output

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_channel_shuffle_nc_x32 failed!");

  const xnn_status run_status = xnn_run_operator(
      channel_shuffle_op,       // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output_padded_contig_nhwc.contiguous(input.suggest_memory_format());
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
