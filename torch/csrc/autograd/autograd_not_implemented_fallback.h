#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace autograd {

TORCH_API void autogradNotImplementedFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack);

}} // namespace torch::autograd