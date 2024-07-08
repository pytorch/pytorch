// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchRulesHelper.h>

namespace at::functorch {

typedef std::tuple<Tensor, optional<int64_t>> oneOutput;
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>> twoOutputs;
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>> threeOutputs;
typedef std::tuple<Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>, Tensor, optional<int64_t>> fourOutputs;

namespace {

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
Tensor vdot_decomp(const Tensor& A, const Tensor& B) {
  return at::dot(A.is_complex() ? A.conj() : A, B);
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

Tensor addmm_decomp(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  // Decomposition that is probably not very fast...
  return at::add(self * beta, at::mm(mat1, mat2), alpha);
}

void _linalg_check_errors_batch_rule(const Tensor& info, optional<int64_t> info_bdim, c10::string_view api_name, bool is_matrix) {
  auto info_ = moveBatchDimToFront(info, info_bdim);
  // Not a matrix means this is a batch of matrices
  at::_linalg_check_errors(info_, api_name, false);
}

std::tuple<Tensor, std::optional<int64_t>>
householder_product_batch_rule(const Tensor &input, std::optional<int64_t> input_bdim,
                               const Tensor &tau, std::optional<int64_t> tau_bdim)
{
  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto tau_ = moveBatchDimToFront(tau, tau_bdim);

  auto batch_size = get_bdim_size2(input, input_bdim, tau, tau_bdim);

  input_ = ensure_has_bdim(input_, input_bdim.has_value(), batch_size);
  tau_ = ensure_has_bdim(tau_, tau_bdim.has_value(), batch_size);
  return std::make_tuple(at::linalg_householder_product(input_, tau_), 0);
}

template <char const *op_name, typename A, A a, typename C>
struct LinalgCheckMatrixUnaryRuleHelper;

template <char const *op_name, typename F, F Func, typename A, typename... T>
struct LinalgCheckMatrixUnaryRuleHelper<op_name, F, Func, typelist<A, T...>> {
  static inline Tensor check_and_reshape_input(const Tensor& tensor, optional<int64_t> batch_dim) {
    TORCH_CHECK(rankWithoutBatchDim(tensor, batch_dim) >= 2, op_name, ": The input tensor A must have at least 2 dimensions.");
    return moveBatchDimToFront(tensor, batch_dim);
  }

  static oneOutput apply_one(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    return std::make_tuple(Func(tensor_, std::forward<T>(extra_args)...), 0);
  }

  static twoOutputs apply_two(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0);
  }

  static threeOutputs apply_three(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
  }

  static fourOutputs apply_four(
      const Tensor& tensor,
      optional<int64_t> batch_dim,
      T... extra_args) {
    const auto tensor_ = check_and_reshape_input(tensor, batch_dim);
    const auto res = Func(tensor_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0, std::get<3>(res), 0);
  }
};

template <char const *op_name, typename A, A a, typename C>
struct LinalgCheckMatrixBinaryRuleHelper;

template <char const *op_name, typename F, F Func, typename A, typename B, typename... T>
struct LinalgCheckMatrixBinaryRuleHelper<op_name, F, Func, typelist<A, B, T...>> {
  static inline std::tuple<Tensor, Tensor> check_inputs_and_reshape_inputs(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim) {
    TORCH_CHECK(rankWithoutBatchDim(first, first_bdim) >= 2,
                op_name, ": The input tensor A must have at least 2 dimensions.");
    TORCH_CHECK(rankWithoutBatchDim(second, second_bdim) >= 2,
                op_name, ": The input tensor B must have at least 2 dimensions.");
    return _binary_pointwise_helper(first, first_bdim, second, second_bdim, false);
  }

  static oneOutput apply_one(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim,
      T... extra_args) {
    const auto tensor_other = check_inputs_and_reshape_inputs(first, first_bdim, second, second_bdim);
    const auto tensor_ = std::get<0>(tensor_other);
    const auto other_ = std::get<1>(tensor_other);
    return std::make_tuple(Func(tensor_, other_, std::forward<T>(extra_args)...), 0);
  }

  static twoOutputs apply_two(
      const Tensor& first, optional<int64_t> first_bdim,
      const Tensor& second, optional<int64_t> second_bdim,
      T... extra_args) {
    const auto tensor_other = check_inputs_and_reshape_inputs(first, first_bdim, second, second_bdim);
    const auto tensor_ = std::get<0>(tensor_other);
    const auto other_ = std::get<1>(tensor_other);
    const auto res = Func(tensor_, other_, std::forward<T>(extra_args)...);
    return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0);
  }
};

static void expect_at_least_rank(
    const Tensor& tensor,
    optional<int64_t> tensor_bdim,
    int64_t expected_rank,
    const char* name) {
  auto rank = rankWithoutBatchDim(tensor, tensor_bdim);
  TORCH_CHECK(rank >= expected_rank,
      name, " should have at least ", expected_rank, " dimensions, but has ",
      rank, " dimensions instead.");
}

threeOutputs linalg_lu_unpack_batch_rule(
    const Tensor& LU, optional<int64_t> LU_bdim,
    const Tensor& pivots, optional<int64_t> pivots_bdim,
    bool unpack_data, bool unpack_pivots) {
  auto LU_ = moveBatchDimToFront(LU, LU_bdim);
  auto pivots_ = moveBatchDimToFront(pivots, pivots_bdim);

  // LU and pivots's first {N-2} (for LU), {N-1} (for pivots) dimensions must
  // match So if only one of them is being vmapped over, we must expand out that
  // dimension.
  if (LU_bdim.has_value() != pivots_bdim.has_value()) {
    auto bdim_size = get_bdim_size2(LU, LU_bdim, pivots, pivots_bdim);
    LU_ = ensure_has_bdim(LU_, LU_bdim.has_value(), bdim_size);
    pivots_ = ensure_has_bdim(pivots_, pivots_bdim.has_value(), bdim_size);
    pivots_bdim = 0;
    LU_bdim = 0;
  }

  const auto res = at::lu_unpack(LU_, pivots_, unpack_data, unpack_pivots);
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
}

oneOutput linalg_lu_solve_batch_rule(
    const Tensor& LU, optional<int64_t> LU_bdim,
    const Tensor& pivots, optional<int64_t> pivots_bdim,
    const Tensor& B, optional<int64_t> B_bdim,
    bool left, bool adjoint) {
  const auto LU_min_rank = 2;
  const auto pivots_min_rank = 1;
  const auto B_min_rank = 2;

  expect_at_least_rank(LU, LU_bdim, LU_min_rank, "LU");
  expect_at_least_rank(pivots, pivots_bdim, pivots_min_rank, "pivots");
  expect_at_least_rank(B, B_bdim, B_min_rank, "B");

  auto LU_ = moveBatchDimToFront(LU, LU_bdim);
  auto pivots_ = moveBatchDimToFront(pivots, pivots_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);

  // LU and pivots's first {N-2} (for LU), {N-1} (for pivots) dimensions must match
  // So if only one of them is being vmapped over, we must expand out that dimension.
  if (LU_bdim.has_value() ^ pivots_bdim.has_value()) {
    auto bdim_size = get_bdim_size2(LU, LU_bdim, pivots, pivots_bdim);
    LU_ = ensure_has_bdim(LU_, LU_bdim.has_value(), bdim_size);
    pivots_ = ensure_has_bdim(pivots_, pivots_bdim.has_value(), bdim_size);
    pivots_bdim = 0;
    LU_bdim = 0;
  }

  // Now, {LU, pivots} and B's first dimensions are allowed to broadcast.
  // The rest of the logic handles that.
  const auto LU_num_batch_dims = rankWithoutBatchDim(LU_, LU_bdim) - LU_min_rank;
  const auto pivots_num_batch_dims = rankWithoutBatchDim(pivots_, pivots_bdim) - pivots_min_rank;
  const auto B_num_batch_dims = rankWithoutBatchDim(B_, B_bdim) - B_min_rank;
  const auto max_num_batch_dims = std::max(std::max(LU_num_batch_dims, pivots_num_batch_dims), B_num_batch_dims);

  LU_ = maybePadToLogicalRank(LU_, LU_bdim, max_num_batch_dims + LU_min_rank);
  pivots_ = maybePadToLogicalRank(pivots_, pivots_bdim, max_num_batch_dims + pivots_min_rank);
  B_ = maybePadToLogicalRank(B_, B_bdim, max_num_batch_dims + B_min_rank);

  const auto result = at::linalg_lu_solve(LU_, pivots_, B_, left, adjoint);
  return std::make_tuple(result, 0);
}

oneOutput cholesky_solve_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim,
    const Tensor& A, std::optional<int64_t> A_bdim,
    bool upper) {
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  TORCH_CHECK(rankWithoutBatchDim(A, A_bdim) >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");

  const auto tensor_other = _binary_pointwise_helper(self, self_bdim, A, A_bdim, /*do_type_promotion=*/false);
  const auto tensor_ = std::get<0>(tensor_other);
  const auto other_ = std::get<1>(tensor_other);
  return std::make_tuple(at::cholesky_solve(tensor_, other_, upper), 0);
}

threeOutputs linalg_lu_factor_ex_batch_rule(
    const Tensor& A, std::optional<int64_t> A_bdim, bool pivot, bool check_errors) {
  TORCH_CHECK(rankWithoutBatchDim(A, A_bdim) >= 2, "torch.lu_factor_ex: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");
  const auto A_ = moveBatchDimToFront(A, A_bdim);
  const auto res = at::linalg_lu_factor_ex(A_, pivot, check_errors);
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0);
}

oneOutput matrix_exp_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim) {
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2, "torch.matrix_exp: The input tensor A must have at least 2 dimensions.");
  const auto self_ = moveBatchDimToFront(self, self_bdim).contiguous();  // seems to be a bug
  return std::make_tuple(at::matrix_exp(self_), 0);
}

fourOutputs solve_ex_batch_rule(
    const Tensor& A, optional<int64_t> A_bdim,
    const Tensor& B, optional<int64_t> B_bdim,
    bool left, bool check_errors) {
  auto batch_size = get_bdim_size2(A, A_bdim, B, B_bdim);
  const auto A_logical_rank = rankWithoutBatchDim(A, A_bdim);
  const auto B_logical_rank = rankWithoutBatchDim(B, B_bdim);
  const auto max_logical_rank = std::max(A_logical_rank, B_logical_rank);

  TORCH_CHECK(A_logical_rank >= 2,
            "linalg.solve: The input tensor A must have at least 2 dimensions.");

  auto b_logical_rank = max_logical_rank;
  if (A_logical_rank > B_logical_rank) {  // vector case: B was a vector or batched vector
    // not accurate but matches linalg error message
    TORCH_CHECK(B_logical_rank >= 1, "linalg.solve: The input tensor B must have at least 2 dimensions.");
    b_logical_rank = max_logical_rank - 1;
  } else {  // matrix case: A and B are both matrices or batches of matrices
    TORCH_CHECK(B_logical_rank >= 2, "linalg.solve: The input tensor B must have at least 2 dimensions.");
  }

  // basically binary pointwise helper but if B was a vector incoming, we must pad it to be 1 dim smaller than A
  auto A_ = moveBatchDimToFront(A, A_bdim);
  auto B_ = moveBatchDimToFront(B, B_bdim);
  A_ = maybePadToLogicalRank(A_, A_bdim, max_logical_rank);
  B_ = maybePadToLogicalRank(B_, B_bdim, b_logical_rank);

  A_ = ensure_has_bdim(A_, A_bdim.has_value(), batch_size);
  B_ = ensure_has_bdim(B_, B_bdim.has_value(), batch_size);

  // NOTE [ solve_ex Batch Rule Contiguity ]
  // A determines whether or not linalg_solve takes an optimized path. We need the check on A_ to match the one run on
  // A as BatchedTensor since it might have been saved by autograd (specifically by the jvp) and the autograd behvaior
  // differs based on whether or not the optimized path was taken
  const auto batched_A_was_contiguous = A_bdim.has_value() ? at::select(A, *A_bdim, 0).is_contiguous() : A.is_contiguous();
  if (batched_A_was_contiguous && !A.is_complex()) {
    A_ = A_.contiguous();
  }
  const auto res = _linalg_solve_ex(A_, B_, left, check_errors);
  return std::make_tuple(std::get<0>(res), 0, std::get<1>(res), 0, std::get<2>(res), 0, std::get<3>(res), 0);
}

oneOutput cross_batch_rule(const Tensor& self, std::optional<int64_t> self_bdim,
                           const Tensor& other, std::optional<int64_t> other_bdim, const int64_t dim) {
  // match cross dimension checks
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) == rankWithoutBatchDim(other, other_bdim),
    "linalg.cross: inputs must have the same number of dimensions."
  );

  const auto batch_size = get_bdim_size2(self, self_bdim, other, other_bdim);
  const auto self_other_bundled = _binary_pointwise_helper(self, self_bdim, other, other_bdim, false);

  const auto self_ = ensure_has_bdim(std::get<0>(self_other_bundled), self_bdim.has_value(), batch_size);
  const auto other_ = ensure_has_bdim(std::get<1>(self_other_bundled), other_bdim.has_value(), batch_size);

  const auto dim_ = getPhysicalDim(self_, true, dim);

  return std::make_tuple(linalg_cross(self_, other_, dim_), 0);
}

std::optional<int64_t> batch_dim_if_not_empty(const Tensor& t) {
  if (t.dim() == 1 && t.size(0) == 0) {
    return std::optional<int64_t>();
  }
  return std::optional<int64_t>(0);
}

fourOutputs linalg_lstsq_batch_rule(
    const Tensor& self, std::optional<int64_t> self_bdim, const Tensor& b, std::optional<int64_t> b_bdim,
    std::optional<double> rcond, std::optional<c10::string_view> driver) {
  TORCH_CHECK(rankWithoutBatchDim(self, self_bdim) >= 2, "torch.linalg.lstsq: input must have at least 2 dimensions.");
  TORCH_CHECK(rankWithoutBatchDim(b, b_bdim) >= 1, "torch.linalg.lstsq: other must have at least 1 dimension.");

  const auto batch_size = get_bdim_size2(self, self_bdim, b, b_bdim);
  const auto tensor_other = _binary_pointwise_helper(self, self_bdim, b, b_bdim, /*do_type_promotion=*/false);

  // because of ambiguity with vector case, lstsq can broadcast [1, 2] -> [batch_size, 2] but not [2] -> [batch_size, 2]
  // so could unsqueeze if there's no bdim or just ensure_has_bdim
  const auto self_ = ensure_has_bdim(std::get<0>(tensor_other), self_bdim.has_value(), batch_size);
  const auto b_ = ensure_has_bdim(std::get<1>(tensor_other), b_bdim.has_value(), batch_size);

  auto [res, res_1, res_2, res_3] = at::linalg_lstsq(self_, b_, rcond, driver);

  // everything but the 0th output are only sometimes computed. When they aren't, they're empty tensors without a bdim
  const auto res_1_bdim = batch_dim_if_not_empty(res_1);
  const auto res_2_bdim = batch_dim_if_not_empty(res_2);
  const auto res_3_bdim = batch_dim_if_not_empty(res_3);
  return std::make_tuple(res, 0, res_1, res_1_bdim, res_2, res_2_bdim, res_3, res_3_bdim);
}

template<typename F>
std::tuple<Tensor, std::optional<int64_t>>
atol_rtol_tensor_batch_rule(
    F Func, const Tensor& input, optional<int64_t> input_bdim,
    const optional<Tensor>& atol, const optional<int64_t> atol_bdim,
    const optional<Tensor>& rtol, const optional<int64_t> rtol_bdim, bool hermitian, char const *op_name) {
  auto input_logical_rank = rankWithoutBatchDim(input, input_bdim);

  TORCH_CHECK(input_logical_rank >= 2,
            op_name, ": The input tensor input must have at least 2 dimensions.");

  // atol and rtol's dims must be broadcastable to the number of batch dims of input
  // which is input's dim - 2 (input represents a batch of matrices, so 2 is for the matrix dimensions)
  const auto input_logical_num_bdims = input_logical_rank - 2;
  const int64_t atol_logical_num_bdims = atol.has_value() ? rankWithoutBatchDim(*atol, atol_bdim) : 0;
  const int64_t rtol_logical_num_bdims = rtol.has_value() ? rankWithoutBatchDim(*rtol, rtol_bdim) : 0;
  const auto max_logical_bdims = std::max({input_logical_num_bdims, atol_logical_num_bdims, rtol_logical_num_bdims});

  auto input_ = moveBatchDimToFront(input, input_bdim);
  auto atol_ = atol.has_value() ? moveBatchDimToFront(*atol, atol_bdim) : atol;
  auto rtol_ = rtol.has_value() ? moveBatchDimToFront(*rtol, rtol_bdim) : rtol;

  // pad all inputs to have the same number of (non-vmap) batch dimensions
  input_ = maybePadToLogicalRank(input_, input_bdim, max_logical_bdims + 2);
  atol_ = atol_.has_value() ? maybePadToLogicalRank(*atol_, atol_bdim, max_logical_bdims) : atol_;
  rtol_ = rtol_.has_value() ? maybePadToLogicalRank(*rtol_, rtol_bdim, max_logical_bdims) : rtol_;

  return std::make_tuple(Func(input_, atol_, rtol_, hermitian), 0);
}

static std::tuple<Tensor, std::optional<int64_t>>
pinv_batch_rule(
    const Tensor& input, std::optional<int64_t> input_bdim, const optional<Tensor>& atol,
    const std::optional<int64_t> atol_bdim, const optional<Tensor>& rtol,
    const std::optional<int64_t> rtol_bdim, bool hermitian) {
  return atol_rtol_tensor_batch_rule(ATEN_FN2(linalg_pinv, atol_rtol_tensor), input, input_bdim, atol, atol_bdim, rtol, rtol_bdim, hermitian, "linalg.pinv");
}
}

#define LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, num_out) SINGLE_ARG(\
  LinalgCheckMatrixUnaryRuleHelper<\
    func_string_##fn,\
    decltype(&ATEN_FN(fn)),\
    &ATEN_FN(fn),\
    c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply_##num_out)

#define LINALG_CHECK_MATRIX_UNARY_BATCH_RULE2(fn, overload, num_out) SINGLE_ARG(\
  LinalgCheckMatrixUnaryRuleHelper<\
    func_string_##fn_##overload,\
    decltype(&ATEN_FN2(fn, overload)),\
    &ATEN_FN2(fn, overload),\
    c10::guts::function_traits<decltype(ATEN_FN2(fn, overload))>::parameter_types>::apply_##num_out)

#define LINALG_CHECK_MATRIX_BINARY_BATCH_RULE(fn, num_out) SINGLE_ARG(\
  LinalgCheckMatrixBinaryRuleHelper<\
    func_string_##fn,\
    decltype(&ATEN_FN(fn)),\
    &ATEN_FN(fn),\
    c10::guts::function_traits<decltype(ATEN_FN(fn))>::parameter_types>::apply_##num_out)


// Define string constants with the function names. These will be used as template parameters
// C++ doesn't let us use string literals as template parameters, so we have to declare them as consts first
// What is going on with these macros?
// - clang-5 seems to require the constexpr
// - windows compiles with or without the constexpr, but the constexpr causes test problems
// - as a result we have some macro guards.
#if defined(_MSC_VER)
#define LINALG_STRING_CONST(fn, op_name) \
  const char func_string_##fn[] = #op_name;\

#define LINALG_STRING_CONST2(fn, overload, op_name) \
  const char func_string_##fn_##overload[] = #op_name;\

#else
#define LINALG_STRING_CONST(fn, op_name) \
  constexpr const char func_string_##fn[] = #op_name;\

#define LINALG_STRING_CONST2(fn, overload, op_name) \
  constexpr const char func_string_##fn_##overload[] = #op_name;\

#endif

#define LINALG_CHECK_MATRIX_UNARY_ONE_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, one));\
  }

#define LINALG_CHECK_MATRIX_UNARY_ONE_OUT2(fn, overload, op_name) \
  LINALG_STRING_CONST2(fn, overload, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT2(fn, overload, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE2(fn, overload, one));\
  }

#define LINALG_CHECK_MATRIX_UNARY_TWO_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, two));\
  }

#define LINALG_CHECK_MATRIX_UNARY_THREE_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, three));\
  }

#define LINALG_CHECK_MATRIX_UNARY_FOUR_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_UNARY_BATCH_RULE(fn, four));\
  }

#define LINALG_CHECK_MATRIX_BINARY_ONE_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_BINARY_BATCH_RULE(fn, one));\
  }

#define LINALG_CHECK_MATRIX_BINARY_TWO_OUT(fn, op_name) \
  LINALG_STRING_CONST(fn, op_name);\
  TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {\
    VMAP_SUPPORT(fn, LINALG_CHECK_MATRIX_BINARY_BATCH_RULE(fn, two));\
  }

// These need to be outside. String constant must be declared outside of a macro to be used as template param
// NOLINTBEGIN(*array*)
LINALG_CHECK_MATRIX_UNARY_ONE_OUT(cholesky, cholesky);
LINALG_CHECK_MATRIX_UNARY_ONE_OUT(cholesky_inverse, cholesky_inverse);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_cholesky_ex, linalg.cholesky);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_eig, linalg.eig);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_inv_ex, linalg.inv_ex);
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(linalg_ldl_factor_ex, torch.linalg.ldl_factor_ex);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_qr, linalg.qr);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(linalg_slogdet, linalg.slogdet);
LINALG_CHECK_MATRIX_BINARY_ONE_OUT(linalg_solve_triangular, linalg.solve_triangular);

LINALG_CHECK_MATRIX_UNARY_TWO_OUT(geqrf, geqrf);
LINALG_CHECK_MATRIX_BINARY_TWO_OUT(triangular_solve, triangular_solve);
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(_linalg_det, linalg.det);
LINALG_CHECK_MATRIX_UNARY_TWO_OUT(_linalg_eigh, linalg.eigh);
LINALG_CHECK_MATRIX_UNARY_FOUR_OUT(_linalg_slogdet, linalg.slogdet);
LINALG_CHECK_MATRIX_UNARY_THREE_OUT(_linalg_svd, linalg.svd);
// NOLINTEND(*array*)

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  VMAP_SUPPORT(bmm, bmm_batch_rule);
  m.impl("addmv", addmv_decomp);
  m.impl("addmm", addmm_decomp);
  m.impl("addbmm", addbmm_decomp);
  m.impl("baddbmm", baddbmm_decomp);
  VMAP_SUPPORT(dot, dot_batch_rule);
  VMAP_SUPPORT(mv, mv_batch_rule);
  VMAP_SUPPORT(mm, mm_batch_rule);
  VMAP_SUPPORT(lu_unpack, linalg_lu_unpack_batch_rule);
  VMAP_SUPPORT(linalg_lu_solve, linalg_lu_solve_batch_rule);
  VMAP_SUPPORT(linalg_householder_product, householder_product_batch_rule);
  VMAP_SUPPORT(cholesky_solve, cholesky_solve_batch_rule);  // custom dim error
  VMAP_SUPPORT(linalg_lstsq, linalg_lstsq_batch_rule);  // custom errors and sometimes empty return
  VMAP_SUPPORT(linalg_lu_factor_ex, linalg_lu_factor_ex_batch_rule);
  VMAP_SUPPORT(linalg_matrix_exp, matrix_exp_batch_rule);
  VMAP_SUPPORT(_linalg_solve_ex, solve_ex_batch_rule);
  VMAP_SUPPORT(linalg_cross, cross_batch_rule);
  VMAP_SUPPORT2(linalg_pinv, atol_rtol_tensor, pinv_batch_rule);

  VMAP_SUPPORT(_linalg_check_errors, _linalg_check_errors_batch_rule);

  m.impl("vdot", vdot_decomp);
}
} // namespace at::functorch
