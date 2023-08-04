// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace at::functorch {

// This file contains code for the vmap fallback (also known as the
// BatchedTensor fallback or the Batched fallback). This code runs
// when an operation doesn't have a batching rule implemented.

// If an operator doesn't have a batching rule implemented then we fallback
// to this implementation. The fallback doesn't work on out= variants or
// view operations; that is, it works for out-of-place operations and
// in-place non-view operations.
//
// For out-of-place operations, the fallback effectively takes all of the
// BatchedTensors in `stack`, slices them, and runs `op` on all of the
// corresponding slices to produce slices of the outputs. The output slices
// then get `torch.stack`ed to create the
// final returns.
//
// The performance of the fallback is not very good because it introduces an
// extra copy from stacking the sliced outputs. Because of this, we prefer to
// write batching rules for operators whenever possible.
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

void vmapErrorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

// The vmap fallback emits a warning by default, but it may be disabled if
// the user finds it to be too annoying.
TORCH_API bool isVmapFallbackWarningEnabled();
TORCH_API void setVmapFallbackWarningEnabled(bool enabled);

// Used for testing. The vmap fallback is enabled by default. When it is disabled,
// it raises an error.
TORCH_API bool isVmapFallbackEnabled();
TORCH_API void setVmapFallbackEnabled(bool enabled);

template <typename A> A vector_to_result(const std::vector<IValue>& buffer) {
  return buffer[0].to<A>();
}
template <typename A, typename B> std::tuple<A, B> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>());
}
template <typename A, typename B, typename C> std::tuple<A, B, C> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>(), buffer[2].to<B>());
}

// slow_fallback is a way to call the vmap fallback inside some boxed kernel.
// There is probably some better way to metaprogram this.
template <typename Ret>
Ret slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  std::vector<IValue> stack(args.begin(), args.end());
  batchedTensorForLoopFallback(op, &stack);
  return vector_to_result<Ret>(stack);
}

template <typename A, typename B>
std::tuple<A, B> slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  std::vector<IValue> stack(args.begin(), args.end());
  batchedTensorForLoopFallback(op, &stack);
  return vector_to_result<A, B>(stack);
}

template <typename A, typename B, typename C>
std::tuple<A, B, C> slow_fallback(const c10::OperatorHandle& op, ArrayRef<IValue> args) {
  std::vector<IValue> stack(args.begin(), args.end());
  batchedTensorForLoopFallback(op, &stack);
  return vector_to_result<A, B, C>(stack);
}


} // namespace at::functorch
