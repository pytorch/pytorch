// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/PlumbingHelper.h>

namespace at::functorch {

void vmap_check_escaped(const std::optional<DynamicLayer> &layer, const char* what) {
  TORCH_CHECK(
    layer.has_value(),
    "Either your tensor may have escaped from inside a function being vmapped and this is a user error ",
    "(see https://pytorch.org/functorch/stable/ux_limitations.html), "
    "or there is an internal functorch error in `",
    what,
    "` Please file an issue if it looks like the latter"
  )
}

Tensor makeBatched(Tensor tensor, std::optional<int64_t> bdim, int64_t level) {
  if (bdim.has_value()) {
    TORCH_INTERNAL_ASSERT(*bdim >= 0);
    TORCH_INTERNAL_ASSERT(*bdim < tensor.dim());
    return makeBatched(std::move(tensor), bdim.value(), level);
  }
  return tensor;
}

std::vector<Tensor> makeBatchedVector(std::vector<Tensor> tensors, std::optional<int64_t> bdim, int64_t level) {
  std::vector<Tensor> res;
  res.reserve(tensors.size());
  for (auto & tensor : tensors) {
    res.emplace_back(makeBatched(std::move(tensor), bdim, level));
  }
  return res;
}

std::tuple<Tensor, std::optional<int64_t>> unwrapTensorAtLevel(const Tensor& tensor, int64_t level) {
  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return std::make_tuple(tensor, std::nullopt);
  }
  if (batched->level() == level) {
    return std::make_tuple(batched->value(), batched->bdim());
  }
  return std::make_tuple(tensor, std::nullopt);
}

bool isBatchedAtLevel(const Tensor& tensor, int64_t level) {
  auto result = unwrapTensorAtLevel(tensor, level);
  return std::get<1>(result).has_value();
}

bool isBatchedAtLevel(const std::optional<Tensor>& maybe_tensor, int64_t level) {
  if (!maybe_tensor.has_value()) {
    return false;
  }
  return isBatchedAtLevel(*maybe_tensor, level);
}

bool isBatchedAtLevel(ITensorListRef tensors, int64_t level) {
  for (const auto& tensor : tensors) {
    if (isBatchedAtLevel(tensor, level)) {
      return true;
    }
  }
  return false;
}

bool isBatchedAtLevel(const c10::List<std::optional<Tensor>>& maybe_tensors, int64_t level) {
  for (const auto idx : c10::irange(0, maybe_tensors.size())) {
    const auto& maybe_tensor = maybe_tensors.get(idx);
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  return false;
}

bool areAnyBatchedAtLevel(ArrayRef<std::optional<Tensor>> maybe_tensors, int64_t level) {
  for (const auto& maybe_tensor : maybe_tensors) {
    if (isBatchedAtLevel(maybe_tensor, level)) {
      return true;
    }
  }
  return false;
}


} // namespace at::functorch
