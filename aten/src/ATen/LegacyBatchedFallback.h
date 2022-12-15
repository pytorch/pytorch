#pragma once
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace at {

// If an operator doesn't have a batching rule implemented then we fallback
// to this implementation. The fallback only works on out-of-place operators
// that return only tensors with new memory. (e.g., no in-place operators, no
// view operations).
//
// The fallback effectively takes all of the BatchedTensors in `stack`, slices
// them, and runs `op` on all of the corresponding slices to produce slices
// of the outputs. The output slices then get `torch.stack`ed to create the
// final returns.
//
// The performance of the fallback is not very good because it introduces an
// extra copy from stacking the sliced outputs. Because of this, we prefer to
// write batching rules for operators whenever possible.
void batchedTensorForLoopFallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

} // namespace at
