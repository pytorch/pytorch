#pragma once

#include <ATen/native/CPUFallback.h>

namespace torch_lazy_tensors {

void ltc_eager_fallback(const c10::OperatorHandle& op,
                        torch::jit::Stack* stack);

}  // namespace torch_lazy_tensors
