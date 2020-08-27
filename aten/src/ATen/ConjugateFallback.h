#pragma once
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace at {

    // This fallback effectively takes all tensors in the stack
    // with their conjugate bit set, and runs conjugation on them
    void conjugateFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

}
