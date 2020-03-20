#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>
#include <ATen/native/xnnpack/Linear.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace linear {

namespace {

// Supports NHWC and NCHW FP32 linear operators.

// TODO: Decouple and improve error handling and messages.
bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::internal::available() &&
          // Weight
          (2 == weight.ndimension()) &&
          (c10::DeviceType::CPU == weight.device().type()) &&
          (kFloat == weight.scalar_type()) &&
          // Bias
          ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                       (c10::DeviceType::CPU == bias->device().type()) &&
                                       (kFloat == bias->scalar_type()) &&
                                       (weight.size(Layout::Filter::output)) == bias->size(0))
                                     : true) &&
          // Output Min / Max
          (output_max > output_min) &&
          true;
}

// TODO: Decouple and improve error handling and messages.
bool usable(const Tensor& input) {
         // Input
  return (2 <= input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type()) &&
         true;
}

Tensor run(
    const ContextLinear& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor padded_input = allocate_padded_if_needed(input.contiguous());

  TORCH_CHECK(
      usable(padded_input),
      "XNNPACK Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  const IntArrayRef input_size = padded_input.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  output_size.back() = context.output_channels;

  Tensor output = empty_with_tail_padding(
      output_size,
      padded_input.options().dtype(),
      padded_input.suggest_memory_format());

  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.op.get(),                                   // operator
      Layout::ActivationND::batch(padded_input.sizes()),  // Batch,
      padded_input.data_ptr<float>(),                     // input
      output.data_ptr<float>(),                           // output
      caffe2::xnnpack_threadpool());                      // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.op.get(),               // operator
      caffe2::xnnpack_threadpool());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output;
}

Tensor create_and_run(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const float output_min,
    const float output_max) {
  return run(
      create(
          weight,
          bias,
          output_min,
          output_max),
      input);
}

} // namespace

ContextLinear create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const float output_min,
    const float output_max) {
  const Tensor weight_contig = weight.contiguous();

  TORCH_CHECK(
      available(
          weight_contig,
          bias,
          output_min,
          output_max),
      "XNNPACK Linear not available! "
      "Reason: The provided (weight, bias, output_min, output_max) parameters are "
      "either invalid individually or their combination is not supported by XNNPACK.");

  xnn_operator_t linear_op{};

  const xnn_status create_status = xnn_create_fully_connected_nc_f32(
      weight_contig.size(Layout::Filter::input),                        // input_channels
      weight_contig.size(Layout::Filter::output),                       // output_channels
      weight_contig.size(Layout::Filter::input),                        // input_pixel_stride
      weight_contig.size(Layout::Filter::output),                       // output_pixel_stride
      weight_contig.data_ptr<float>(),                                  // kernel
      (bias && bias->defined()) ? bias->data_ptr<float>() : nullptr,  // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &linear_op);                                                    // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_fully_connected_nc_f32 failed!");

  return ContextLinear(
    Operator(linear_op),
    weight_contig.size(Layout::Filter::output)
  );
}

c10::intrusive_ptr<xnnpack::XNNPackLinearOpContext>
    LinearPrePack::operator()(
        Tensor weight,
        c10::optional<Tensor> bias) {
      return xnnpack::XNNPackLinearOpContext::create_context(
          std::move(weight), std::move(bias), {}, {});
}

Tensor LinearPacked::operator()(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::XNNPackLinearOpContext>& op_context) {
      return
        xnnpack::internal::linear::run((op_context.get())->get_context(), input);
}

} // namespace linear
} // namespace internal

bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  return internal::linear::available(
            weight,
            bias,
            ContextLinear::kMin,
            ContextLinear::kMax) &&
         internal::linear::usable(input);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  return internal::linear::create_and_run(
      input,
      weight,
      bias,
      ContextLinear::kMin,
      ContextLinear::kMax);
}

} // namespace xnnpack

} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
