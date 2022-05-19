//  Copyright © 2022 Apple Inc.

#include <ATen/native/CPUFallback.h>

namespace at {

void mps_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
  TORCH_WARN_ONCE("The operator '", op.schema().operator_name(), "' is not currently supported ",
                  "on the MPS backend and will fall back to run on the CPU.",
                  " This may performance implications.");
  native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, MPS, m) {
  static const char *enable_mps_fallback = getenv("PYTORCH_ENABLE_MPS_FALLBACK");
  if (!enable_mps_fallback || std::stoi(enable_mps_fallback) == 0) {
    TORCH_WARN("The op is currently not supported on MPS. You can use",
    "`PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable to enable fallback on",
    " CPU. This may have perf implications.");
    return;
  }
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  // These ops are not supported via MPS backend currently, and we fallback to run on CPU.
  // For the rest of unsupported ops the user needs to pass 'PYTORCH_ENABLE_MPS_FALLBACK=1'
  // to fallback on CPU, otherwise we will error out.
  m.impl("bitwise_and.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_or.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_xor.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_not.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_left_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_right_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("embedding_renorm_", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd.U", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_fft_c2c", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_fft_r2c", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_vector_norm", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("sgn.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nonzero", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("masked_select", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
}

} // namespace at
