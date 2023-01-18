// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <ATen/Tensor.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>

// NOTE: [vmap plumbing]
//
// Here's how "batching rules" work.
// - we register kernels to the Batched key
// - these kernels have the same signatures as the original operators.
//   For example, at::sin(Tensor self) accepts a Tensor, and the batched kernel
//   must also accept a Tensor
// - However, it is more natural for users to write a batching rule like the
//   following: sin_batch_rule(Tensor self, optional<int> self_bdim)
// - There is some codegenerated layer (the "plumbing") that wraps the user
//   defined batching rule (e.g. sin_batch_rule) in a kernel that can be
//   registered to the Batched key.
//
// The plumbing is responsible for wrapping a batching rule into a form that may
// be registered as the kernel for the batched key.

namespace at { namespace functorch {

void vmap_check_escaped(const optional<DynamicLayer> &layer, const char* what);

// Create a BatchedTensor given a tensor, bdim, and level
TORCH_API Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level);

// Given a Tensor that may or may not be a BatchedTensor, unwrap it.
// If `tensor` is not a BatchedTensor, or is a BatchedTensor but the level
// doesn't match, then this returns (tensor, nullopt).
// Otherwise, it returns (unwrap(tensor), bdim).
TORCH_API std::tuple<Tensor, c10::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level);

// Creates a vector of BatchedTensor
TORCH_API std::vector<Tensor> makeBatchedVector(const std::vector<Tensor>& tensors, optional<int64_t> bdim, int64_t level);

// Returns True if ANY tensor in tensors is batched at level
TORCH_API bool isBatchedAtLevel(ITensorListRef tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const c10::List<c10::optional<Tensor>> maybe_tensors, int64_t level);
TORCH_API bool isBatchedAtLevel(const Tensor& tensor, int64_t level);
TORCH_API bool isBatchedAtLevel(const c10::optional<Tensor>& maybe_tensor, int64_t level);

// Convenience helper. Returns true if any tensor is batched at level
TORCH_API bool areAnyBatchedAtLevel(ArrayRef<optional<Tensor>> maybe_tensors, int64_t level);

inline bool ivalueParticipatesInCurrentLevel(const IValue& ivalue) {
  if (ivalue.isTensor()) {
    auto maybe_level = maybeCurrentDynamicLayer();
    TORCH_INTERNAL_ASSERT(maybe_level.has_value());
    auto current_level = maybe_level->layerId();
    return isBatchedAtLevel(ivalue.toTensor(), current_level);
  }
  // TODO: should really check this
  return false;
}

}}
