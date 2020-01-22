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
namespace clamp {

struct Context final {
  Operator clamp_op;
};

namespace {

// Supports NHWC and NCHW FP32
//  - relu
//  - relu6
//  - leaky relu
//  - hard tanh
//  - or any other operator that can be expressed as a clamp

bool available(
    const int64_t input_channels,
    const int64_t output_channels,
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
         // Output Min / Max
         (output_min.isIntegral(true) || output_min.isFloatingPoint()) &&
         (output_max.isIntegral(true) || output_max.isFloatingPoint()) &&
         (output_max.to<float>() > output_min.to<float>());
}

Context create(
    const int64_t input_channels,
    const int64_t output_channels,
    const Scalar output_min,
    const Scalar output_max) {
  TORCH_INTERNAL_ASSERT(
      available(input_channels, output_channels, output_min, output_max),
      "mobile::cpu::clamp not available!");

  xnn_operator_t clamp_op{};

  const xnn_status create_status = xnn_create_clamp_nc_f32(
      input_channels,         // channels,
      input_channels,         // input_pixel_stride
      output_channels,        // output_pixel_stride
      output_min.to<float>(), // output_min
      output_max.to<float>(), // output_max
      0u,                     // flags
      &clamp_op);             // operator

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == create_status,
      "xnn_create_clamp_nc_f32 failed!");

  return Context{
    Operator(clamp_op),
  };
}

bool usable(const Tensor& input) {
         // Input
  return (0 < input.ndimension()) &&
         (c10::DeviceType::CPU == input.device().type()) &&
         (kFloat == input.scalar_type());
}

Tensor& run(
    const Context& context,
    Tensor& output,
    const Tensor& input) {
  using namespace mobile::internal;

  TORCH_INTERNAL_ASSERT(
      usable(input),
      "mobile::cpu::clamp not usable!");

  const xnn_status setup_status = xnn_setup_clamp_nc_f32(
      context.clamp_op.get(),                     // operator
      Layout::ActivationND::Batch(input.sizes()), // Batch
      input.data_ptr<float>(),                    // input
      output.data_ptr<float>(),                   // output
      threadpool().handle());                     // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == setup_status,
      "xnn_setup_clamp_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.clamp_op.get(), // operator
      threadpool().handle()); // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output;
}

Tensor output(const Tensor& input) {
  return internal::new_tensor(
      input.sizes(),
      input.options(),
      input.suggest_memory_format());
}

Tensor& create_and_run(
    Tensor& output,
    const Tensor& input,
    const Scalar output_min,
    const Scalar output_max) {
  using namespace mobile::internal;

  return run(
      create(
          Layout::ActivationND::Channel(input.sizes()),
          Layout::ActivationND::Channel(output.sizes()),
          output_min,
          output_max),
      output,
      input);
}

Tensor create_and_run(
    const Tensor& input,
    const Scalar output_min,
    const Scalar output_max) {
  Tensor output = clamp::output(input);

  return create_and_run(
      output,
      input,
      output_min,
      output_max);
}

const auto registry = c10::RegisterOperators()
  .op("mobile::clamp_create",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const int64_t input_channels,
          const int64_t output_channels,
          const Scalar output_min,
          const Scalar output_max) {
        return cpp_custom_type_hack::create(
            std::make_unique<Context>(
                create(
                    input_channels,
                    output_channels,
                    output_min,
                    output_max)),
            c10::TensorOptions{});
      }))
  .op("mobile::clamp_run",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const c10::optional<Tensor>& context,
          c10::optional<Tensor> output,
          const Tensor& input) {
        TORCH_INTERNAL_ASSERT(context, "Invalid context!");

        if (output && output->defined()) {
          return run(
              cpp_custom_type_hack::cast<Context>(*context),
              *output,
              input);
        }

        Tensor output_ = clamp::output(input);

        return run(
            cpp_custom_type_hack::cast<Context>(*context),
            output_,
            input);
      }))
  .op("mobile::clamp",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          c10::optional<Tensor> output,
          const Tensor& input,
          const Scalar output_min,
          const Scalar output_max) {
        if (output && output->defined()) {
          return create_and_run(
              *output,
              input,
              output_min,
              output_max);
        }

        return create_and_run(
            input,
            output_min,
            output_max);
      }));

} // namespace
} // namespace clamp
} // namespace internal

bool use_clamp(
    const Tensor& input,
    const Scalar output_min,
    const Scalar output_max) {
  using namespace internal;

  const int64_t input_channels = Layout::ActivationND::Channel(input.sizes());
  const int64_t output_channels = input_channels;

  return internal::clamp::available(
            input_channels,
            output_channels,
            output_min,
            output_max) &&
         internal::clamp::usable(input);
}

Tensor& clamp(
    Tensor& output,
    const Tensor& input,
    const Scalar output_min,
    const Scalar output_max) {
  using namespace internal;

  return internal::clamp::create_and_run(
      output,
      input,
      output_min,
      output_max);
}

Tensor clamp(
    const Tensor& input,
    const Scalar output_min,
    const Scalar output_max) {
  return internal::clamp::create_and_run(
      input,
      output_min,
      output_max);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::mobile::cpu::internal::clamp::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
