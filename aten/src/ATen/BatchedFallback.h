#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace at {

void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace at
