// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>
#include <ATen/functorch/PlumbingHelper.h>
#include <ATen/Operators.h>

// NB: most activation functions fit pointwise unary or binary rules.
// These are only the ones that have special batch rules to help with organization
namespace at::functorch {
static std::tuple<Tensor,optional<int64_t>>
glu_batch_rule(const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  // repeated error message from glu because 0D -> 1D when batched
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");

  const auto rank = rankWithoutBatchDim(self, self_bdim);
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  const auto self_ = moveBatchDimToFront(self, self_bdim);

  const auto res = at::glu(self_, dim_);
  return std::make_tuple(res, 0);
}

static std::tuple<Tensor,optional<int64_t>> glu_backward_batch_rule(
    const Tensor& grad_output, optional<int64_t> grad_output_bdim,
    const Tensor& self, optional<int64_t> self_bdim, int64_t dim) {
  if (self_bdim) {
    // repeated error message from glu because 0D -> 1D when batched
    // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
    // can't be evenly halved, but give a nicer error message here.
    TORCH_CHECK(self.dim() > 1, "glu does not support 0-dimensional tensors");
  }

  const auto rank = rankWithoutBatchDim(self, self_bdim);
  const auto dim_ = maybe_wrap_dim(dim, rank) + 1;

  const auto batch_size = get_bdim_size2(grad_output, grad_output_bdim, self, self_bdim);
  const auto grad_output_ = ensure_has_bdim(moveBatchDimToFront(grad_output, grad_output_bdim), grad_output_bdim.has_value(), batch_size);
  const auto self_ = ensure_has_bdim(moveBatchDimToFront(self, self_bdim), self_bdim.has_value(), batch_size);

  const auto res = at::glu_backward(grad_output_, self_, dim_);
  return std::make_tuple(res, 0);
}


TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(glu_backward, glu_backward_batch_rule);
  VMAP_SUPPORT(glu, glu_batch_rule);
}
} // namespace at::functorch
