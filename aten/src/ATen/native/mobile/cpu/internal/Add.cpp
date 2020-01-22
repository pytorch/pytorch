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
namespace add {

struct Context final {
  Operator add_op;

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

namespace {

// Supports NHWC and NCHW FP32 adds.

bool available(
    const int64_t input1_channels,
    const int64_t input2_channels,
    const int64_t output_channels,
    const Scalar output_min,
    const Scalar output_max) {
         // Mobile
  return mobile::cpu::available() &&
         // Input1
         (input1_channels > 0) &&
         // Input2
         (input2_channels > 0) &&
         // Input1 == Input2
         (input1_channels == input2_channels) &&
         // Output
         (output_channels > 0) &&
         // Input == Output
         (input1_channels == output_channels) &&
         // Output Min / Max
         (output_min.isIntegral(true) || output_min.isFloatingPoint()) &&
         (output_max.isIntegral(true) || output_max.isFloatingPoint()) &&
         (output_max.to<float>() > output_min.to<float>());
}

Context create(
    const int64_t input1_channels,
    const int64_t input2_channels,
    const int64_t output_channels,
    const Scalar output_min,
    const Scalar output_max) {
  TORCH_INTERNAL_ASSERT(
      available(input1_channels, input2_channels, output_channels, output_min, output_max),
      "mobile::cpu::add not available!");

  xnn_operator_t add_op{};

  const xnn_status create_status = xnn_create_add_nc_f32(
      output_channels,        // channels
      input1_channels,        // a_stride
      input2_channels,        // b_stride
      output_channels,        // sum_stride
      output_min.to<float>(), // sum_min
      output_max.to<float>(), // sum_max
      0u,                     // flags
      &add_op);               // operator

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == create_status,
      "xnn_create_add_nc_f32 failed!");

  return Context{
    Operator(add_op),
  };
}

bool usable(
    const Tensor& input1,
    const Tensor& input2) {
         // Input1
  return (0 < input1.ndimension()) &&
         (c10::DeviceType::CPU == input1.device().type()) &&
         (kFloat == input1.scalar_type()) &&
         // Input2
         (0 < input2.ndimension()) &&
         (c10::DeviceType::CPU == input2.device().type()) &&
         (kFloat == input2.scalar_type()) &&
         // Input1 (Batch, Channel) == Input2 (Batch, Channel)
         (Layout::ActivationND::Batch(input1.sizes()) ==
            Layout::ActivationND::Batch(input2.sizes())) &&
         (Layout::ActivationND::Channel(input1.sizes()) ==
            Layout::ActivationND::Channel(input2.sizes())) &&
         // Same Memory Layout
         (input1.suggest_memory_format() ==
            input2.suggest_memory_format());
}

Tensor& run(
    const Context& context,
    Tensor& output,
    const Tensor& input1,
    const Tensor& input2) {
  using namespace mobile::internal;

  TORCH_INTERNAL_ASSERT(
      usable(input1, input2),
      "mobile::cpu::add not usable!");

  const xnn_status setup_status = xnn_setup_add_nc_f32(
      context.add_op.get(),                         // operator
      Layout::ActivationND::Batch(input1.sizes()),  // Batch
      input1.data_ptr<float>(),                     // a
      input2.data_ptr<float>(),                     // b
      output.data_ptr<float>(),                     // output
      threadpool().handle());                       // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == setup_status,
      "xnn_setup_add_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
      context.add_op.get(),   // operator
      threadpool().handle()); // threadpool

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  return output;
}

Tensor output(
    const Tensor& input1,
    const Tensor& input2) {
  const Tensor& input = (input1.ndimension() < input2.ndimension()) ?
                        input1 :
                        input2;

  return internal::new_tensor(
      input.sizes(),
      input.options(),
      input.suggest_memory_format());
}

Tensor& create_and_run(
    Tensor& output,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar output_min,
    const Scalar output_max)
{
  using namespace mobile::internal;

  return run(
      create(
          Layout::ActivationND::Channel(input1.sizes()),
          Layout::ActivationND::Channel(input2.sizes()),
          Layout::ActivationND::Channel(output.sizes()),
          output_min,
          output_max),
      output,
      input1,
      input2);
}

Tensor create_and_run(
    const Tensor& input1,
    const Tensor& input2,
    const Scalar output_min,
    const Scalar output_max)
{
  Tensor output = add::output(
      input1,
      input2);

  return create_and_run(
      output,
      input1,
      input2,
      output_min,
      output_max);
}

const auto registry = c10::RegisterOperators()
  .op("mobile::add_create",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const int64_t input1_channels,
          const int64_t input2_channels,
          const int64_t output_channels,
          const Scalar output_min,
          const Scalar output_max) {
        return cpp_custom_type_hack::create(
            std::make_unique<Context>(
                create(
                    input1_channels,
                    input2_channels,
                    output_channels,
                    output_min,
                    output_max)),
            c10::TensorOptions{});
      }))
  .op("mobile::add_run",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          const c10::optional<Tensor>& context,
          c10::optional<Tensor> output,
          const Tensor& input1,
          const Tensor& input2) {
        TORCH_INTERNAL_ASSERT(context, "Invalid context!");

        if (output && output->defined()) {
          return run(
              cpp_custom_type_hack::cast<Context>(*context),
              *output,
              input1,
              input2);
        }

        Tensor output_ = add::output(
            input1,
            input2);

        return run(
              cpp_custom_type_hack::cast<Context>(*context),
              output_,
              input1,
              input2);

      }))
  .op("mobile::add",
    c10::RegisterOperators::options()
      .kernel(DispatchKey::CPUTensorId, [](
          c10::optional<Tensor> output,
          const Tensor& input1,
          const Tensor& input2,
          const Scalar output_min,
          const Scalar output_max) {
        if (output && output->defined()) {
          return create_and_run(
              *output,
              input1,
              input2,
              output_min,
              output_max);
        }

        return create_and_run(
            input1,
            input2,
            output_min,
            output_max);
      }));

} // namespace
} // namespace add
} // namespace internal

bool use_add(
    const Tensor& input1,
    const Tensor& input2) {
  using namespace internal;

  const int64_t input1_channels = Layout::ActivationND::Channel(input1.sizes());
  const int64_t input2_channels = Layout::ActivationND::Channel(input2.sizes());
  const int64_t output_channels = std::min(input1_channels, input2_channels);

  return internal::add::available(
            input1_channels,
            input2_channels,
            output_channels,
            add::Context::kMin,
            add::Context::kMax) &&
         internal::add::usable(input1, input2);
}

Tensor& add(
    Tensor& output,
    const Tensor& input1,
    const Tensor& input2) {
  return internal::add::create_and_run(
      output,
      input1,
      input2,
      internal::add::Context::kMin,
      internal::add::Context::kMax);
}

Tensor add(
    const Tensor& input1,
    const Tensor& input2)
{
  return internal::add::create_and_run(
      input1,
      input2,
      internal::add::Context::kMin,
      internal::add::Context::kMax);
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

namespace caffe2 {

CAFFE_KNOWN_TYPE(at::native::mobile::cpu::internal::add::Context);

} // namespace caffe2

#endif /* USE_XNNPACK */
