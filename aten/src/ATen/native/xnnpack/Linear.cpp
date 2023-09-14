#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Linear.h>

namespace at::native::xnnpack {
namespace internal::linear {

namespace {

// Supports NHWC and NCHW FP32 linear operators.

// TODO: Decouple and improve error handling and messages.
bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const float output_min,
    const float output_max) {
         // XNNPACK
  return xnnpack::available() &&
          // Weight
          (2 == weight.ndimension()) &&
          (weight.device().is_cpu()) &&
          (kFloat == weight.scalar_type()) &&
          !weight.requires_grad() &&
          // Bias
          ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                       (bias->device().is_cpu()) &&
                                       (kFloat == bias->scalar_type()) &&
                                       (weight.size(Layout::Filter::output)) == bias->size(0) &&
                                       !bias->requires_grad())
                                     : true) &&
          // Output Min / Max
          (output_max > output_min) &&
          true;
}

// TODO: Decouple and improve error handling and messages.
bool usable(const Tensor& input) {
         // Input
  return (1 <= input.ndimension()) &&
         (input.device().is_cpu()) &&
         (kFloat == input.scalar_type()) &&
         !input.requires_grad() &&
         true;
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

} // anonymous namespace

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
      (bias && bias->defined()) ?
          bias->contiguous().data_ptr<float>() :
          nullptr,                                                      // bias
      output_min,                                                     // output_min
      output_max,                                                     // output_max
      0u,                                                             // flags
      nullptr,                                                        // xnn_caches_t
      &linear_op);                                                    // operator

  TORCH_CHECK(
      xnn_status_success == create_status,
      "xnn_create_fully_connected_nc_f32 failed!");

  return ContextLinear(
    Operator(linear_op),
    weight_contig.size(Layout::Filter::output)
  );
}

Tensor run(
    const ContextLinear& context,
    const Tensor& input) {
  using namespace internal;

  // For compatibility with aten::linear
  auto ip = input;
  if (input.ndimension() == 1) {
    ip = input.unsqueeze(0);
  }

  const Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
      ip, ip.suggest_memory_format());

  TORCH_CHECK(
      usable(padded_input),
      "XNNPACK Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  const IntArrayRef input_size = padded_input.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  output_size.back() = context.output_channels;

  Tensor output = mobile::empty_with_tail_padding(
      output_size,
      padded_input.options().dtype(),
      padded_input.suggest_memory_format(),
      padded_input.opt_names());

  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.op.get(),                                   // operator
      Layout::ActivationND::batch(padded_input.sizes()),  // Batch,
      padded_input.data_ptr<float>(),                     // input
      output.data_ptr<float>(),                           // output
      caffe2::pthreadpool_());                            // threadpool

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.op.get(),         // operator
      caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  // For compatibility with aten::linear
  if (input.ndimension() == 1) {
      output.squeeze_(0);
  }

  return output;
}

c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max) {
  return xnnpack::XNNPackLinearOpContext::create_context(
      std::move(weight), std::move(bias), output_min, output_max);
}

Tensor linear_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context) {
  return op_context->run(input);
}

IValue
unpack_prepacked_sizes_linear(const IValue& ivalue) {
  auto op_context = ivalue.toCustomClass<xnnpack::LinearOpContext>();
  const auto tuple = op_context->unpack();
  const auto& bias = std::get<1>(tuple);
  return IValue(std::make_tuple(
      std::get<0>(tuple).sizes(),
      (bias && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : c10::nullopt));
}

} // namespace internal::linear

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

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
