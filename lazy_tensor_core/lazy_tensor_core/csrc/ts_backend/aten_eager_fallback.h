#pragma once

#include <functional>

#include <ATen/native/CPUFallback.h>

namespace torch_lazy_tensors {

bool force_eager_fallback(c10::Symbol op);
void ltc_eager_fallback(const c10::OperatorHandle& op,
                        torch::jit::Stack* stack);

extern std::function<void(void)> register_ts_ltc_eager_fallback;

}  // namespace torch_lazy_tensors
