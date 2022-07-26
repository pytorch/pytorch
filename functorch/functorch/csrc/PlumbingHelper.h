// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <ATen/Tensor.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/Constants.h>
#include <functorch/csrc/DynamicLayer.h>

namespace at { namespace functorch {

Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level);
std::tuple<Tensor, optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level);

std::vector<Tensor> makeBatchedVector(const std::vector<Tensor>& tensors, optional<int64_t> bdim, int64_t level);

// Returns True if ANY tensor in tensors is batched at level
bool isBatchedAtLevel(TensorList tensors, int64_t level);
bool isBatchedAtLevel(const c10::List<c10::optional<Tensor>> maybe_tensors, int64_t level);
bool isBatchedAtLevel(const Tensor& tensor, int64_t level);
bool isBatchedAtLevel(const c10::optional<Tensor>& maybe_tensor, int64_t level);

// Convenience helper. Returns true if any tensor is batched at level
bool areAnyBatchedAtLevel(ArrayRef<optional<Tensor>> maybe_tensors, int64_t level);

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
