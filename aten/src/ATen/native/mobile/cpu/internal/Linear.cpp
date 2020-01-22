#include <ATen/native/mobile/cpu/Engine.h>

#ifdef USE_XNNPACK

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/mobile/cpu/internal/Allocator.h>
#include <ATen/native/mobile/cpu/internal/Common.h>
#include <ATen/native/mobile/internal/ThreadPool.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {
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

bool available(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const Scalar output_min,
    const Scalar output_max) {
         // Mobile
  return mobile::cpu::available() &&
         // Weight
         (2 == weight.ndimension()) &&
         (c10::DeviceType::CPU == weight.device().type()) &&
         (kFloat == weight.scalar_type()) &&
         // Bias
         ((bias && bias->defined()) ? ((1 == bias->ndimension()) &&
                                      (c10::DeviceType::CPU == bias->device().type()) &&
                                      (kFloat == bias->scalar_type()) &&
                                      (weight.size(Layout::Filter::Output)) == bias->size(0))
                                    : true) &&
         // Output Min / Max
         (output_min.isIntegral(true) || output_min.isFloatingPoint()) &&
         (output_max.isIntegral(true) || output_max.isFloatingPoint()) &&
         (output_max.to<float>() > output_min.to<float>());
}

Context create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const Scalar output_min,
    const Scalar output_max) {
  TORCH_INTERNAL_ASSERT(
      available(
          weight,
          bias,
          output_min,
          output_max),
      "mobile::cpu::linear not available!");

  xnn_operator_t linear_op{};

  const xnn_status create_status = xnn_create_fully_connected_nc_f32(
      weight.size(Layout::Filter::Input),                               // input_channels
      weight.size(Layout::Filter::Output),                              // output_channels
      weight.size(Layout::Filter::Input),                               // input_pixel_stride
      weight.size(Layout::Filter::Output),                              // output_pixel_stride
      weight.data_ptr<float>(),                                         // kernel
      (bias && bias->defined()) ? bias->data_ptr<float>() : nullptr,    // bias
      output_min.to<float>(),                                           // output_min
      output_max.to<float>(),                                           // output_max
      0u,                                                               // flags
      &linear_op);                                                      // operator

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == create_status,
      "xnn_create_fully_connected_nc_f32 failed!");

  return Context{
    Operator(linear_op),
    {
      weight.size(Layout::Filter::Output),
    }
  };
}

bool usable(const Tensor& input) {
         // Input
  return (2 <= input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type());
}

Tensor run(
    const Context& context,
    const Tensor& input) {
  using namespace mobile::internal;

  TORCH_INTERNAL_ASSERT(
      usable(input),
      "mobile::cpu::linear not usable!");

  const IntArrayRef input_size = input.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  output_size.back() = context.output.channels;

  Tensor output = new_tensor(
      output_size,
      input.options(),
      input.suggest_memory_format());

  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.linear_op.get(),                    // operator
      Layout::ActivationND::Batch(input.sizes()), // Batch,
      input.data_ptr<float>(),                    // input
      output.data_ptr<float>(),                   // output
      threadpool().handle());                     // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.linear_op.get(),  // operator
      threadpool().handle());   // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output;
}

Tensor create_and_run(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Scalar output_min,
    const Scalar output_max) {
  return run(
      create(
          weight,
          bias,
          output_min,
          output_max),
      input);
}

const auto registry = c10::RegisterOperators()
  .op("mobile::linear_create",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const Tensor& weight,
          const c10::optional<Tensor>& bias,
          const Scalar output_min,
          const Scalar output_max) {
        return cpp_custom_type_hack::create(
            std::make_unique<Context>(
                create(
                    weight,
                    bias,
                    output_min,
                    output_max)),
            weight.options());
      }))
  .op("mobile::linear_run",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const Tensor& context,
          const Tensor& input) {
        return run(
            cpp_custom_type_hack::cast<Context>(context),
            input);
      }));

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

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::mobile::cpu::internal::linear::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
