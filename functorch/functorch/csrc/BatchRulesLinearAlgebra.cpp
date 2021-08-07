// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>>
slogdet_batch_rule(const Tensor& self, optional<int64_t> self_bdim) {
  // slogdet supports arbitrary dims at the front
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::slogdet(self_);
  return std::make_tuple(
    std::move(std::get<0>(result)), 0,
    std::move(std::get<1>(result)), 0
  );
}

// Note [Batching rules for matmul-like operators]
// at::matmul doesn't "de-expand" arguments to get better performance (maybe
// it should). In the batching rules for matmul-like operators (dot, mv, mm),
// we should be careful not to expand any unnecessary dimensions. i.e., if
// only one of the two arguments is a BatchedTensor, then we should try
// not to expand batch dimensions onto the other arg.

std::tuple<Tensor, optional<int64_t>> dot_batch_rule(const Tensor& A, optional<int64_t> A_bdim, const Tensor& B, optional<int64_t> B_bdim) {
  TORCH_CHECK(A.dim() - A_bdim.has_value() == 1 && B.dim() - B_bdim.has_value() == 1, "Got wrong shapes for dot");
  auto A_ = moveBatchDimToFront(A, A_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);
  if (A_bdim && B_bdim) {
    return std::make_tuple(at::matmul(A_.unsqueeze(-2), B_.unsqueeze(-1)).squeeze(-1).squeeze(-1), 0);
  } else {
    return std::make_tuple(at::matmul(A_, B_.t()), 0);
  }
}

// NB: I wrote this like this because we *might* want its for a future matmul
// batch rule that isn't decomposed...
// "tv" = tensor @ vector
static std::tuple<Tensor, optional<int64_t>> tv_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  if (self_bdim && other_bdim) {
    // See Note [Batching rules for matmul-like operators]
    // B...OI, BI -> ...BOI, BI1 -> ...BO1 -> ...BO
    auto self_ = at::movedim(self, *self_bdim, -3);
    auto other_ = moveBatchDimToFront(other, other_bdim);
    other_ = other_.unsqueeze(-1);
    auto result = at::matmul(self_, other_).squeeze(-1);
    auto result_bdim = result.dim() - 2;
    return std::make_tuple( std::move(result), result_bdim );
  }
  else if (self_bdim && !other_bdim) {
    // B...OI, I -> B...O
    auto self_ = moveBatchDimToFront(self, self_bdim);
    return std::make_tuple( at::matmul(self_, other), 0 );
  }
  else if (!self_bdim && other_bdim) {
    // ...OI, BI -> ...OI, IB -> OB
    auto other_ = at::movedim(other, *other_bdim, -1);
    auto result = at::matmul(self, other_);
    return std::make_tuple( std::move(result), 1 );
  }
  TORCH_INTERNAL_ASSERT(false, "can't get here");
}

static std::tuple<Tensor, optional<int64_t>> mv_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  TORCH_CHECK(self_logical_rank == 2 && other_logical_rank == 1,
      "Shape mismatch: ",
      "Got incorrect dims for mv(a, b). a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 2 and dim 1");
  return tv_batch_rule(self, self_bdim, other, other_bdim);
}

static std::tuple<Tensor, optional<int64_t>> mm_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  TORCH_CHECK(self_logical_rank == 2 && other_logical_rank == 2,
      "Shape mismatch: Got incorrect dims for mm(a, b). "
      "a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 2 and dim 2");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);
  return std::make_tuple( at::matmul(self_, other_), 0 );
}

static std::tuple<Tensor, optional<int64_t>> bmm_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim,
    const Tensor& other, optional<int64_t> other_bdim) {
  auto self_logical_rank = rankWithoutBatchDim(self, self_bdim);
  auto other_logical_rank = rankWithoutBatchDim(other, other_bdim);
  TORCH_CHECK(self_logical_rank == 3 && other_logical_rank == 3,
      "Shape mismatch: Got incorrect dims for bmm(a, b). "
      "a has dim ", self_logical_rank,
      "and b has dim ", other_logical_rank,
      "but expected them to have dim 3 and dim 3");
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto other_ = moveBatchDimToFront(other, other_bdim);
  return std::make_tuple( at::matmul(self_, other_), 0 );
}

Tensor linear_decomp(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias_opt) {
  auto result = input.matmul(weight.t());
  if (bias_opt) {
    // NB: It's too much work to figure out how to actually fuse the bias so
    // we're not going to.
    // TODO: if the result isn't batched but bias is, then we need to do the following.
    // Otherwise, it can just be in-place. We should write a more nuanced
    // decomposition rule
    return result.add(*bias_opt);
  }
  return result;
}

std::tuple<Tensor,optional<int64_t>,Tensor,optional<int64_t>> linalg_eigh_batch_rule(
    const Tensor& self, optional<int64_t> self_bdim, c10::string_view UPLO) {
  auto self_ = moveBatchDimToFront(self, self_bdim);
  auto result = at::linalg_eigh(self_, UPLO);
  return std::make_tuple(std::get<0>(result), 0, std::get<1>(result), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("bmm", bmm_batch_rule);
  VMAP_SUPPORT("dot", dot_batch_rule);
  VMAP_SUPPORT("mv", mv_batch_rule);
  VMAP_SUPPORT("mm", mm_batch_rule);
  VMAP_SUPPORT("linalg_eigh", linalg_eigh_batch_rule);
  VMAP_SUPPORT("slogdet", slogdet_batch_rule);
  OP_DECOMPOSE(einsum);
  OP_DECOMPOSE(outer);
  OP_DECOMPOSE(ger);
  m.impl("linear", linear_decomp);
  VMAP_SUPPORT("cholesky", VARIADIC_BDIMS_BATCH_RULE(at::cholesky));
  VMAP_SUPPORT("cholesky_inverse", VARIADIC_BDIMS_BATCH_RULE(at::cholesky_inverse));
  VMAP_SUPPORT("pinverse", VARIADIC_BDIMS_BATCH_RULE(at::pinverse));
  VMAP_SUPPORT("linalg_pinv", VARIADIC_BDIMS_BATCH_RULE(ATEN_FN(linalg_pinv)));
}
}}

