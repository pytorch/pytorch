#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>

namespace at {
namespace native {
namespace xnnpack {

// Supports NHWC and NCHW FP32
//  - relu
//  - relu6
//  - hard tanh
//  - or any other operator that can be expressed as a clamp

bool use_clamp(
    const Tensor& input,
    const float output_min,
    const float output_max) {
  using namespace internal;

  return xnnpack::internal::available() &&
      // Input
      (input.dim() > 0) &&
      (Layout::ActivationND::channel(input.sizes()) > 0) &&
      (c10::DeviceType::CPU == input.device().type()) &&
      (kFloat == input.scalar_type()) &&
      !input.requires_grad() &&
      // Output Min / Max
      (output_max > output_min) &&
      true;
}

bool use_clamp_(
    Tensor& input,
    const float output_min,
    const float output_max) {
  using namespace internal;

      // In place clamp.  Input / output pre-allocated and have no control over it.
  return can_avoid_reallocation(input, input.suggest_memory_format()) &&
      // Clamp
      use_clamp(input, output_min, output_max);
}

bool use_clamp_out(
    Tensor& output,
    const Tensor& input,
    const float output_min,
    const float output_max) {
  using namespace internal;

      // Output pre-allocated and have no control over it, but can still reallocate input.
  return can_avoid_reallocation(output, output.suggest_memory_format()) &&
      // Clamp
      use_clamp(input, output_min, output_max);
}

namespace internal {
namespace {

Tensor& clamp(
    Tensor& output,
    const Tensor& input,
    const float output_min,
    const float output_max) {
  const IntArrayRef input_sizes = input.sizes();
  const size_t batches = Layout::ActivationND::batch(input_sizes);
  const size_t channels = Layout::ActivationND::channel(input_sizes);

  xnn_operator_t clamp_op{};

  const xnn_status create_status = xnn_create_clamp_nc_f32(
      channels,   // channels,
      channels,   // input_pixel_stride - Contiguous
      channels,   // output_pixel_stride - Contiguous
      output_min, // output_min
      output_max, // output_max
      0u,         // flags
      &clamp_op); // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_clamp_nc_f32 failed!");

  const xnn_status setup_status = xnn_setup_clamp_nc_f32(
      clamp_op,                       // operator
      batches,                        // batch_size
      input.data_ptr<float>(),        // input
      output.data_ptr<float>(),       // output
      caffe2::xnnpack_threadpool());  // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_clamp_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      clamp_op,                       // operator
      caffe2::xnnpack_threadpool());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output;
}

} // namespace internal
} // namespace

Tensor clamp(
    const Tensor& input,
    const float output_min,
    const float output_max) {
  using namespace internal;

  const Tensor input_padded_contig = allocate_padded_contiguous_if_needed(
      input,
      input.suggest_memory_format());

  Tensor output_padded_contig = empty_with_tail_padding(
      input_padded_contig.sizes(),
      input_padded_contig.options().dtype(),
      input_padded_contig.suggest_memory_format(),
      input_padded_contig.names());

  return internal::clamp(output_padded_contig, input_padded_contig, output_min, output_max);
}

Tensor& clamp_(
    Tensor& input,
    const float output_min,
    const float output_max) {
  return internal::clamp(input, input, output_min, output_max);
}

Tensor& clamp_out(
    Tensor& output,
    const Tensor & input,
    const float output_min,
    const float output_max) {
  using namespace internal;

  const Tensor input_padded_contig = allocate_padded_contiguous_if_needed(
      input,
      input.suggest_memory_format());

  return internal::clamp(output, input_padded_contig, output_min, output_max);
}

} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
