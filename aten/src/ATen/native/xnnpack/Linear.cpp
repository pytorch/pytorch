#ifdef USE_XNNPACK

#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Factory.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace linear {

struct Context final {
  Operator linear_op;

  struct Output final {
    int64_t channels;
  } output;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

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

Context create(
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
      weight_contig.size(Layout::Filter::input),                      // input_channels
      weight_contig.size(Layout::Filter::output),                     // output_channels
      weight_contig.size(Layout::Filter::input),                      // input_pixel_stride
      weight_contig.size(Layout::Filter::output),                     // output_pixel_stride
      weight_contig.data_ptr<float>(),                                // kernel
      (bias && bias->defined()) ? bias->data_ptr<float>() : nullptr,  // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      &linear_op);                                                    // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_fully_connected_nc_f32 failed!");

  return Context{
    Operator(linear_op),
    {
      weight_contig.size(Layout::Filter::output),
    }
  };
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
    const Context& context,
    const Tensor& input) {
  using namespace internal;

  const Tensor& input_contig = input.contiguous();

  TORCH_CHECK(
      usable(input_contig),
      "XNNPACK Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  const IntArrayRef input_size = input_contig.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  output_size.back() = context.output.channels;

  Tensor output = empty_with_tail_padding(
      output_size,
      input_contig.options().dtype(),
      input_contig.suggest_memory_format());

  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.linear_op.get(),                            // operator
      Layout::ActivationND::batch(input_contig.sizes()),  // Batch,
      input_contig.data_ptr<float>(),                     // input
      output.data_ptr<float>(),                           // output
      nullptr);                                           // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.linear_op.get(),  // operator
      nullptr);                 // threadpool

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
} // namespace linear
} // namespace internal

bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias) {
  return internal::linear::available(
            weight,
            bias,
            internal::linear::Context::kMin,
            internal::linear::Context::kMax) &&
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
      internal::linear::Context::kMin,
      internal::linear::Context::kMax);
}

} // namespace xnnpack

Tensor _linear_prepack(
    const Tensor& weight,
    const Tensor& bias,
    const c10::optional<double> output_min,
    const c10::optional<double> output_max) {
  return cpp_custom_type_hack::create(
      std::make_unique<xnnpack::internal::linear::Context>(
          xnnpack::internal::linear::create(
              weight,
              bias,
              output_min ? *output_min : xnnpack::internal::linear::Context::kMin,
              output_max ? *output_max : xnnpack::internal::linear::Context::kMax)),
      weight.options());
}

Tensor _linear_packed(
    const Tensor& packed_weight,
    const Tensor& input) {
  return xnnpack::internal::linear::run(
      cpp_custom_type_hack::cast<xnnpack::internal::linear::Context>(packed_weight),
      input);
}

} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::xnnpack::internal::linear::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
