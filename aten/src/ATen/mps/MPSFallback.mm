//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/CPUFallback.h>

namespace at {

static void mps_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_WARN_ONCE("The operator '",
                  op.schema().operator_name(),
                  "' is not currently supported ",
                  "on the MPS backend and will fall back to run on the CPU.",
                  " This may have performance implications.");

  auto& profiler = mps::getMPSProfiler();
  const bool isCPUFallbackProfilingEnabled = profiler.isCPUFallbackProfilingEnabled();

  // only do profiling if CPU Fallback op execution tracing or logging is enabled
  if (isCPUFallbackProfilingEnabled) {
    // we create a Tensors list to compute the size of copies required to convert
    // the input MPS tensors to CPU, and the CPU results back to MPS
    std::vector<at::Tensor> tensor_args;
    for (const auto& ivalue : torch::jit::last(stack, op.schema().arguments().size())) {
      if (ivalue.isTensor()) {
        tensor_args.push_back(ivalue.toTensor());
      }
    }
    // TODO: check if any returns exist at this stage
    for (const auto& ivalue : torch::jit::last(stack, op.schema().returns().size())) {
      if (ivalue.isTensor()) {
        tensor_args.push_back(ivalue.toTensor());
      }
    }
    profiler.beginProfileCPUFallback(op.schema().name(), tensor_args);
  }

  native::cpu_fallback(op, stack);

  if (isCPUFallbackProfilingEnabled) {
    profiler.endProfileCPUFallback(op.schema().name());
  }
}

static void mps_error_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "The operator '",
      op.schema().operator_name(),
      "' is not currently implemented ",
      "for the MPS device. If you want this op to be added in priority during the prototype ",
      "phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. ",
      "As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` ",
      "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
      "on MPS.");
}

// This dispatch should never be called for tensor on MPS but is frequently called
// If one of them are on CPU
static Tensor slow_conv2d_forward_mps(const Tensor& self,
                                      const Tensor& weight,
                                      IntArrayRef kernel_size,
                                      const c10::optional<Tensor>& bias,
                                      IntArrayRef stride,
                                      IntArrayRef padding) {
  TORCH_CHECK(self.device() == weight.device(),
              __func__,
              ": input(device='",
              self.device(),
              "') and weight(device=",
              weight.device(),
              "')  must be on the same device");
  TORCH_INTERNAL_ASSERT(false, __func__, " should not be called for both tensors on MPS device");
}

TORCH_LIBRARY_IMPL(_, MPS, m) {
  static const char* enable_mps_fallback = getenv("PYTORCH_ENABLE_MPS_FALLBACK");
  if (!enable_mps_fallback || std::stoi(enable_mps_fallback) == 0) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&mps_error_fallback>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  }
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  // These ops are not supported via MPS backend currently, and we fallback to run on CPU.
  // For the rest of unsupported ops the user needs to pass 'PYTORCH_ENABLE_MPS_FALLBACK=1'
  // to fallback on CPU, otherwise we will error out.
  m.impl("bitwise_left_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_right_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("embedding_renorm_", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd.U", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("im2col", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>()); // Used in  preprocessing by nn.Unfold
  m.impl("col2im", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_slow_conv2d_forward", slow_conv2d_forward_mps);
  m.impl("upsample_nearest3d.vec", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
}

} // namespace at
