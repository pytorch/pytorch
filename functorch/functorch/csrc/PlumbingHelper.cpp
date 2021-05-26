// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/BatchedTensorImpl.h>

namespace at { namespace functorch {

Tensor makeBatched(const Tensor& tensor, optional<int64_t> bdim, int64_t level) {
  if (bdim.has_value()) {
    TORCH_INTERNAL_ASSERT(*bdim >= 0);
    TORCH_INTERNAL_ASSERT(*bdim < tensor.dim());
    return makeBatched(tensor, {{level, bdim.value()}});
  }
  return tensor;
}

std::tuple<Tensor, optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return {tensor, nullopt};
  }
  TORCH_INTERNAL_ASSERT(batched->bdims().size() == 1);
  auto batched_level = batched->bdims().back().level();
  if (batched_level == level) {
    auto bdim = batched->bdims().back().dim();
    return {batched->value(), bdim};
  }
  return {tensor, nullopt};
}

}}
