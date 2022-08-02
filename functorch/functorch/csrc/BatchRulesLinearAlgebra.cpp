// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {

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

// AFAICT, nothing here can be batched. So we decompose :)
Tensor addmv_decomp(
  const Tensor& input, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha) {
  Tensor out = at::mv(mat, vec);
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  return out;
}

Tensor addbmm_decomp(
  const Tensor& input, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor out = at::bmm(batch1, batch2).sum(0);
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  return out;
}

Tensor baddbmm_decomp(
  const Tensor& input, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor out = at::bmm(batch1, batch2);
  if (!alpha.equal(1)) {
    out = alpha * out;
  }
  if (!beta.equal(0)) {
    out = beta * input + out;
  }
  return out;
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

Tensor addmm_decomp(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  // Decomposition that is probably not very fast...
  return at::add(self * beta, at::mm(mat1, mat2), alpha);
}

void _linalg_check_errors_batch_rule(const Tensor& info, optional<int64_t> info_bdim, c10::string_view api_name, bool is_matrix) {
  auto info_ = moveBatchDimToFront(info, info_bdim);
  // Not a matrix means this is a batch of matrices
  at::_linalg_check_errors(info_, api_name, false);
}

std::tuple<Tensor, c10::optional<int64_t>>
householder_product_batch_rule(const Tensor &input, c10::optional<int64_t> input_bdim,
                               const Tensor &tau, c10::optional<int64_t> tau_bdim)
{
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto tau_ = moveBatchDimToFront(tau, tau_bdim);

  auto batch_size = get_bdim_size2(input, input_bdim, tau, tau_bdim);

  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);
  tau_ = ensure_has_bdim(tau_, tau_bdim.has_value(), batch_size);
  return std::make_tuple(at::linalg_householder_product(input_, tau_), 0);
}

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT(bmm, bmm_batch_rule);
  m.impl("addmv", addmv_decomp);
  m.impl("addmm", addmm_decomp);
  m.impl("addbmm", addbmm_decomp);
  m.impl("baddbmm", baddbmm_decomp);
  VMAP_SUPPORT(dot, dot_batch_rule);
  VMAP_SUPPORT(mv, mv_batch_rule);
  VMAP_SUPPORT(mm, mm_batch_rule);
  m.impl("linear", linear_decomp);
  VMAP_SUPPORT(linalg_householder_product, householder_product_batch_rule);

  VMAP_SUPPORT(_linalg_check_errors, _linalg_check_errors_batch_rule);

  VARIADIC_BDIMS_BOXED(cholesky_solve);
  VARIADIC_BDIMS_BOXED(linalg_cholesky_ex);
  VARIADIC_BDIMS_BOXED(linalg_eig);
  VARIADIC_BDIMS_BOXED(linalg_eigh);
  VARIADIC_BDIMS_BOXED(linalg_inv_ex);
  VARIADIC_BDIMS(linalg_pinv);
  VARIADIC_BDIMS_BOXED(linalg_qr);
  VARIADIC_BDIMS_BOXED(linalg_slogdet);

  VARIADIC_BDIMS(cholesky);
  VARIADIC_BDIMS(cholesky_inverse);
  VARIADIC_BDIMS_BOXED(geqrf);
  VARIADIC_BDIMS(logdet);
  VARIADIC_BDIMS(matrix_exp);
  VARIADIC_BDIMS(pinverse);
  VARIADIC_BDIMS(inverse);
  VARIADIC_BDIMS_BOXED(slogdet);
  VARIADIC_BDIMS_BOXED(_linalg_svd);
  VARIADIC_BDIMS_BOXED(solve);
  VARIADIC_BDIMS_BOXED(symeig);
  VARIADIC_BDIMS_BOXED(triangular_solve);

  VARIADIC_BDIMS_BOXED(_linalg_det);
  VARIADIC_BDIMS_BOXED(_lu_with_info);
}
}}
