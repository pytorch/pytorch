// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>

namespace at {
namespace functorch {

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
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

bool isVmapFallbackWarningEnabled();
void setVmapFallbackWarningEnabled(bool enabled);

bool isVmapFallbackEnabled();
void setVmapFallbackEnabled(bool enabled);

template <typename A> A vector_to_result(const std::vector<IValue>& buffer) {
  return buffer[0].to<A>();
}
template <typename A, typename B> std::tuple<A, B> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>());
}
template <typename A, typename B, typename C> std::tuple<A, B, C> vector_to_result(const std::vector<IValue>& buffer) {
  return std::make_tuple(buffer[0].to<A>(), buffer[1].to<B>(), buffer[2].to<B>());
}

// This is a way to call the slow fallback from inside some plumbing
// TODO: Probably better way to metaprogram this
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


}
} // namespace at
