#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/Matmul.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <c10/util/variant.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_compute_linear_combination_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_det.h>
#include <ATen/ops/_linalg_det_native.h>
#include <ATen/ops/_linalg_slogdet.h>
#include <ATen/ops/_linalg_slogdet_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/addbmm_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addr.h>
#include <ATen/ops/addr_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/ceil.h>
#include <ATen/ops/chain_matmul_native.h>
#include <ATen/ops/det_native.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/frobenius_norm_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/ger_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/inner_native.h>
#include <ATen/ops/is_complex_native.h>
#include <ATen/ops/is_floating_point_native.h>
#include <ATen/ops/kron_native.h>
#include <ATen/ops/linalg_cond.h>
#include <ATen/ops/linalg_cond_native.h>
#include <ATen/ops/linalg_det.h>
#include <ATen/ops/linalg_det_native.h>
#include <ATen/ops/linalg_diagonal_native.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_matmul_native.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/linalg_matrix_exp_native.h>
#include <ATen/ops/linalg_matrix_norm.h>
#include <ATen/ops/linalg_matrix_norm_native.h>
#include <ATen/ops/linalg_matrix_power_native.h>
#include <ATen/ops/linalg_matrix_rank.h>
#include <ATen/ops/linalg_matrix_rank_native.h>
#include <ATen/ops/linalg_multi_dot_native.h>
#include <ATen/ops/linalg_norm.h>
#include <ATen/ops/linalg_norm_native.h>
#include <ATen/ops/linalg_pinv.h>
#include <ATen/ops/linalg_pinv_native.h>
#include <ATen/ops/linalg_slogdet.h>
#include <ATen/ops/linalg_slogdet_native.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/linalg_tensorinv_native.h>
#include <ATen/ops/linalg_tensorsolve.h>
#include <ATen/ops/linalg_tensorsolve_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/linalg_vector_norm_native.h>
#include <ATen/ops/log2.h>
#include <ATen/ops/logdet_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/matrix_exp_backward_native.h>
#include <ATen/ops/matrix_exp_native.h>
#include <ATen/ops/matrix_power_native.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/norm.h>
#include <ATen/ops/nuclear_norm_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/outer.h>
#include <ATen/ops/outer_native.h>
#include <ATen/ops/pinverse_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/prod.h>
#include <ATen/ops/real.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/slogdet_native.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/tensordot.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

namespace at {

namespace detail {
  static void check_linalg_norm_dtype(optional<ScalarType> opt_dtype, ScalarType self_dtype, const char* const name) {
    if (opt_dtype.has_value()) {
      auto dtype = opt_dtype.value();
      TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype), name, ": dtype should"
          " be floating point or complex, but got ", dtype);
      TORCH_CHECK(isComplexType(self_dtype) == isComplexType(dtype),
          name, ": dtype should be ", isComplexType(self_dtype) ? "complex" : "real",
          " for ", isComplexType(self_dtype) ? "complex" : "real", " inputs, but got ", dtype);
      TORCH_CHECK(promoteTypes(self_dtype, dtype) == dtype,
          name, ": the dtype of the input ", "(", self_dtype, ") should be convertible ",
          "without narrowing to the specified dtype (", dtype, ")");
    }
  }
}

namespace meta {

#define ADDMM_META() \
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "self and mat2 must have the same dtype"); \
  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "mat1 and mat2 must have the same dtype"); \
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor"); \
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor"); \
  TORCH_CHECK( \
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (", \
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")"); \
 \
  auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self); \
  set_output_raw_strided(0, {mat1.sizes()[0], mat2.sizes()[1]}, {}, mat1.options(), names);

TORCH_META_FUNC(addmm)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  ADDMM_META();
}

TORCH_META_FUNC(_addmm_activation)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu) {
  ADDMM_META();
}

TORCH_META_FUNC(mm)(const Tensor & self, const Tensor & mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x", self.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  auto names = at::namedinference::compute_matmul_outnames(self, mat2);
  set_output_raw_strided(0, {self.sizes()[0], mat2.sizes()[1]}, {}, self.options(), names);
}

TORCH_META_FUNC(linalg_vector_norm)(const Tensor& self, const Scalar& scalar_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  at::native::checkFloatingOrComplex(self, "linalg.vector_norm");

  auto dim = opt_dim.value_or(IntArrayRef{});
  // Casting a large integer to a double will just introduce an error for
  // values larger than 10^53 (same for negative numbers), so that's fine.
  auto ord = scalar_ord.toDouble();

  // For more context, see issue 52783
  // If the tensor is empty and norm < 0 || norm == infty
  //   - We cannot reduce the whole tensor
  //   - We cannot reduce over an empty dimension
  if (self.numel() == 0 && (ord < 0. || ord == INFINITY)) {
    // dim=None or dim=() reduces the whole tensor
    TORCH_CHECK(opt_dim.has_value() && !opt_dim->empty(),
      "linalg.vector_norm cannot compute the ", scalar_ord, " norm on an empty ",
      "tensor because the operation does not have an identity");
    for (auto dim_num : dim) {
      TORCH_CHECK(self.size(dim_num) != 0,
        "linalg.vector_norm cannot compute the ", scalar_ord, " norm on the dimension ", dim_num ,
        "because this dimension is empty and the operation does not have an identity");
    }
  }

  at::detail::check_linalg_norm_dtype(opt_dtype, self.scalar_type(), "linalg.vector_norm");

  auto mask = at::native::make_dim_mask(dim, self.dim());
  auto shape = at::native::shape_from_dim_mask(self, std::move(mask), keepdim);
  auto options = self.options()
                     .dtype(toRealValueType(opt_dtype.value_or(self.scalar_type())));

  set_output_raw_strided(0, shape, {}, options);
}

TORCH_META_FUNC(_linalg_det)(const Tensor& A) {
  at::native::squareCheckInputs(A, "linalg.det");
  at::native::checkFloatingOrComplex(A, "linalg.det");

  auto shape = A.sizes();
  auto ndim = shape.size();

  // det
  set_output_contiguous(0, shape.slice(0, ndim - 2), A.options());

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(1, shape, LU_strides, A.options());

  // pivots
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

TORCH_META_FUNC(_linalg_slogdet)(const Tensor& A) {
  at::native::squareCheckInputs(A, "linalg.slogdet");
  at::native::checkFloatingOrComplex(A, "linalg.slogdet", /*low_precision*/false);

  auto shape= A.sizes();
  auto ndim = shape.size();

  auto shape_outputs = shape.slice(0, ndim - 2);

  // sign
  set_output_contiguous(0, shape_outputs, A.options());

  // logabsdet
  set_output_contiguous(1, shape_outputs, A.options().dtype(toRealValueType(A.scalar_type())));

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(2, shape, LU_strides, A.options());

  // pivots
  set_output_contiguous(3, shape.slice(0, ndim - 1), A.options().dtype(kInt));
}

template <typename Meta>
void common_checks_baddbmm_bmm(Meta& meta, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm, const c10::optional<Tensor>& self_baddbmm = nullopt) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  std::vector<int64_t> output_size {bs, res_rows, res_cols};

  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
              "Expected size for first two dimensions of batch2 tensor to be: [",
              bs, ", ", contraction_size, "] but got: [", batch2_sizes[0], ", ", batch2_sizes[1], "].");

  auto& result = meta.maybe_get_output(0);
  // 'set_output' does not resize for in-place calls
  meta.set_output_raw_strided(0, output_size, {}, batch2.options());
  const auto result_sizes = result.sizes();
  // Error is raised if called from in-place overload with incorrect shape
  TORCH_CHECK(result_sizes == output_size,
              "Expected an output tensor with shape [", output_size, "] but got shape ", result_sizes);

  std::vector<Dimname> outnames = {};
  if (!is_bmm) {
    if (self_baddbmm.has_value()) {
      const auto& self = self_baddbmm.value();
      if (beta.toComplexDouble() != 0.0) result.copy_(self);
      TORCH_CHECK(self.dim() == 3, "self must be a 3D tensor");
      const auto self_sizes = self.sizes();
      TORCH_CHECK(self_sizes == output_size,
                  "Expected an input tensor shape with shape ", output_size, " but got shape: ", self_sizes);
      outnames = namedinference::compute_baddbmm_outnames(result, batch1, batch2, self);
    }
  } else {
    outnames = namedinference::compute_bmm_outnames(result, batch1, batch2);
  }

  namedinference::propagate_names_if_nonempty(
    result,
    outnames
  );
}

TORCH_META_FUNC(bmm)(const Tensor& self, const Tensor& mat2) {
    common_checks_baddbmm_bmm(*this, self, mat2, Scalar(0.0), Scalar(1.0), true);
}

TORCH_META_FUNC(baddbmm)(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  auto self_ = expand_size(self, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  common_checks_baddbmm_bmm(*this, batch1, batch2, beta, alpha, false, *self_);
}

} // namespace meta
namespace native {

DEFINE_DISPATCH(addr_stub);


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.det ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// As P is a permutation matrix
// det(P) = 1 if it's an even permutation and det(P) = -1 if it's an odd permutation
Tensor lu_det_P(const Tensor& pivots) {
  return (at::arange(1, pivots.size(-1) + 1, pivots.options()) != pivots)
    .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong)
    .fmod_(2)
    // take 0 to 1 and 1 to -1
    .mul_(-2)
    .add_(1);
}

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_det_out)(const Tensor& A, const Tensor& result, const Tensor& LU, const Tensor& pivots) {
  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it copies
  // Use the transpose of if A is contiguous since det(A^T) = det(A)
  // We limit this to real matrices, but it could also be implemented for complex matrices
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  // det = det_P * prod(diag(LU))
  at::mul_out(const_cast<Tensor&>(result), lu_det_P(pivots), at::prod(LU.diagonal(0, -2 ,-1), /*dim=*/-1));
}

Tensor linalg_det(const Tensor& A) {
  return std::get<0>(at::_linalg_det(A));
}

Tensor& linalg_det_out(const Tensor& A, Tensor& result) {
  auto LU = at::empty({0}, A.options());
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  at::_linalg_det_out(result, LU, pivots, A);
  return result;
}

// torch.det, alias for torch.linalg.det
Tensor det(const Tensor& self) {
  return at::linalg_det(self);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.slogdet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_slogdet_out)(const Tensor& A, const Tensor& sign, const Tensor& logabsdet, const Tensor& LU, const Tensor& pivots) {
  // info is an aux tensor
  auto info = at::empty({0}, A.options().dtype(kInt));
  // Optimisation: lu_factor_ex requires the input to be F-contig, otherwise it copies
  // Use the transpose of if A is contiguous since det(A^T) = det(A)
  // We limit this to real matrices, but it could also be implemented for complex matrices
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU), const_cast<Tensor&>(pivots), const_cast<Tensor&>(info), A.is_contiguous() && !A.is_complex() ? A.mH() : A);

  auto diag_U = LU.diagonal(0, -2, -1);
  // sign
  at::mul_out(const_cast<Tensor&>(sign), diag_U.sgn().prod(-1), lu_det_P(pivots));

  // logabsdet
  at::sum_out(const_cast<Tensor&>(logabsdet), diag_U.abs().log_(), -1);
}

std::tuple<Tensor, Tensor> linalg_slogdet(const Tensor& A) {
  auto out = at::_linalg_slogdet(A);
  return std::make_tuple(std::move(std::get<0>(out)), std::move(std::get<1>(out)));
}

std::tuple<Tensor&, Tensor&> linalg_slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  auto LU = at::empty({0}, A.options());
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  at::_linalg_slogdet_out(sign, logabsdet, LU, pivots, A);
  return std::tie(sign, logabsdet);
}

// Alias
std::tuple<Tensor, Tensor> slogdet(const Tensor& A) {
  return at::linalg_slogdet(A);
}

std::tuple<Tensor&, Tensor&> slogdet_out(const Tensor& A, Tensor& sign, Tensor& logabsdet) {
  return at::linalg_slogdet_out(sign, logabsdet, A);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ logdet ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor logdet(const Tensor& A) {
  squareCheckInputs(A, "logdet");
  checkFloatingOrComplex(A, "logdet", /*low_precision*/false);
  Tensor sign, logabsdet;
  std::tie(sign, logabsdet) = at::linalg_slogdet(A);

  if (A.is_complex()) {
    return sign.log() + logabsdet;
  } else {
    return at::where(sign == -1., NAN, logabsdet);
  }
}

namespace {

// This function extracts the optional Tensors for atol and rtol
// Default value for atol is zero
// Default value for rtol is eps*max(rows, cols)
// If atol is specified and rtol is not specified then default value for rtol is zero
// It is used for matrix_rank and pinv
std::tuple<Tensor, Tensor> get_atol_rtol(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    const c10::string_view function_name) {
  auto options = input.options().dtype(ScalarType::Double);
  auto atol = atol_opt.has_value() ? atol_opt.value() : at::zeros({}, options);
  checkNotComplexTolerance(atol, function_name, "atol");
  Tensor rtol;
  if (rtol_opt.has_value()) {
    rtol = rtol_opt.value();
    checkNotComplexTolerance(rtol, function_name, "rtol");
  } else {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    auto default_rtol = at::full({}, _get_epsilon(real_dtype) * std::max(input.sym_size(-1), input.sym_size(-2)), options);
    rtol = atol_opt.has_value()
           ? at::where(atol_opt.value() > 0, at::zeros({}, options), default_rtol)
           : std::move(default_rtol);
  }
  return std::make_tuple(atol, rtol);
}

std::tuple<Tensor, Tensor> get_atol_rtol(
    const Tensor& input,
    optional<double> atol_opt,
    optional<double> rtol_opt) {
  double atol = atol_opt.has_value() ? atol_opt.value() : 0.0;
  c10::SymFloat rtol;
  if (rtol_opt.has_value()) {
    rtol = rtol_opt.value();
  } else {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    auto default_rtol = _get_epsilon(real_dtype) * std::max(input.sym_size(-1), input.sym_size(-2));
    rtol = (atol_opt.has_value() && atol_opt.value() > 0.0)
           ? 0.0
           : default_rtol;
  }
  auto options = input.options().dtype(ScalarType::Double);
  auto atol_tensor = at::full({}, atol, options);
  auto rtol_tensor = at::full({}, rtol, options);
  return std::make_tuple(atol_tensor, rtol_tensor);
}

} // anonymous namespace

Tensor linalg_pinv(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    bool hermitian) {
  // FIXME: Whenever we have a nice lstsq, we should dispatch this function to simply be
  // `torch.lstsq(A, torch.eye(A.shape[-1]), atol=atol, rtol=rtol)`
  // with a driver that supports singular inputs
  NoTF32Guard disable_tf32;
  ScalarType t = input.scalar_type();
  TORCH_CHECK((t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble)
              && input.dim() >= 2,
              "linalg.pinv(", t, "{", input.sizes(), "}): expected a tensor with 2 or more dimensions "
              "of float, double, cfloat or cdouble types");

  Tensor atol, rtol;
  std::tie(atol, rtol) = get_atol_rtol(input, atol_opt, rtol_opt, "torch.linalg.pinv");

  if (input.sym_numel() == 0) {
    // The implementation below uses operations that do not work for zero numel tensors
    // therefore we need this early return for 'input.numel() == 0' case
    Tensor U, S, V;
    // TODO: replace input.svd with linalg_svd when torch/xla can work with at::linalg_svd
    std::tie(U, S, V) = input.svd();
    return at::matmul(V * S.reciprocal().unsqueeze(-2), U.mH());
  }

  // If not Hermitian use singular value decomposition, else use eigenvalue decomposition
  if (!hermitian) {
    Tensor U, S, V;
    // TODO: replace input.svd with linalg_svd
    // using linalg_svd breaks pytorch/xla, see https://github.com/pytorch/xla/issues/2755
    std::tie(U, S, V) = input.svd();
    Tensor max_val = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);  // singular values are sorted in descending order
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    Tensor S_pseudoinv = at::where(S > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes V @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(V * S_pseudoinv.unsqueeze(-2), U.mH());
  } else {
    Tensor S, U;
    std::tie(S, U) = at::linalg_eigh(input);
    // For Hermitian matrices, singular values equal to abs(eigenvalues)
    Tensor S_abs = S.abs();
    // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
    Tensor max_val = S_abs.amax(/*dim=*/-1, /*keepdim=*/true);
    Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_val);
    Tensor S_pseudoinv = at::where(S_abs > tol, S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes U @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(U * S_pseudoinv.unsqueeze(-2), U.mH());
  }
}

Tensor linalg_pinv(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian) {
  Tensor atol_tensor, rtol_tensor;
  std::tie(atol_tensor, rtol_tensor) = get_atol_rtol(input, atol, rtol);
  return at::linalg_pinv(input, atol_tensor, rtol_tensor, hermitian);
}

Tensor linalg_pinv(const Tensor& input, const Tensor& rcond, bool hermitian) {
  // For NumPy compatibility the rcond argument is used as relative tolerance
  checkNotComplexTolerance(rcond, "torch.linalg.pinv", "rcond");
  auto options = input.options().dtype(ScalarType::Double);
  return at::linalg_pinv(input, at::zeros({}, options), rcond, hermitian);
}

Tensor linalg_pinv(const Tensor& input, double rcond, bool hermitian) {
  // For NumPy compatibility the rcond argument is used as relative tolerance
  return at::linalg_pinv(input, 0.0, rcond, hermitian);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_pinv_out(
    const Tensor& input,
    const optional<Tensor>& atol,
    const optional<Tensor>& rtol,
    bool hermitian,
    Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(
    const Tensor& input,
    optional<double> atol,
    optional<double> rtol,
    bool hermitian,
    Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);
  Tensor result_tmp = at::linalg_pinv(input, atol, rtol, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(const Tensor& input, const Tensor& rcond, bool hermitian, Tensor& result) {
  checkSameDevice("linalg.pinv", result, input);
  checkLinalgCompatibleDtype("linalg.pinv", result, input);

  Tensor result_tmp = at::linalg_pinv(input, rcond, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(const Tensor& input, double rcond, bool hermitian, Tensor& result) {
  Tensor rcond_tensor = at::full({}, rcond, input.options().dtype(ScalarType::Double));
  return at::linalg_pinv_out(result, input, rcond_tensor, hermitian);
}

Tensor pinverse(const Tensor& self, double rcond) {
  return at::linalg_pinv(self, rcond, /*hermitian=*/false);
}

// matrix_power implementation
namespace {

/**
 * @brief Raises the input matrix to the given power n
 *
 * If the exponent n is negative, the inverse of the input
 * matrix will be raised to power abs(n).
 *
 * @param self (batched) square matrix to raise to power n
 * @param n exponent to raise matrix (or matrices in batch) to
 * @param _out optional tensor to write the output to
 * @return Tensor input matrix raised to power n
 */
Tensor linalg_matrix_power_impl(
    const Tensor& self,
    int64_t n,
    c10::optional<Tensor> _out) {
  NoTF32Guard disable_tf32;
  auto out = _out.value_or(Tensor());

  squareCheckInputs(self, "linalg.matrix_power");
  if (_out.has_value()) {
    checkSameDevice("matrix_power", out, self);
    checkLinalgCompatibleDtype("matrix_power", out, self);
    at::native::resize_output(out, self.sizes());
  }

  // For n=0 we return the identity matrix of the same shape as input.
  if (n == 0) {
    if (!_out.has_value()) {
      // Clone input to include result in the autograd graph
      out = self.clone(at::MemoryFormat::Contiguous);
    }
    return out.copy_(at::eye_symint(self.sym_size(-2), self.options()));
  }
  if (n == 1) {
    return _out.has_value() ? out.copy_(self)
                            : self.clone(at::MemoryFormat::Contiguous);
  }
  if (n == -1) {
    return _out.has_value() ? at::linalg_inv_out(out, self)
                            : at::linalg_inv(self);
  }

  // For negative n we inverte the input matrix before raising to power abs(n)
  auto a = n < 0 ? at::linalg_inv(self) : self;
  n = std::abs(n);

  // Fast paths for small powers
  if (n == 2) {
    return _out.has_value() ? at::matmul_out(out, a, a) : at::matmul(a, a);
  }
  if (n == 3) {
    return _out.has_value() ? at::matmul_out(out, at::matmul(a, a), a)
                            : at::matmul(at::matmul(a, a), a);
  }

  // This is a binary decomposition of n.
  // Moving from the least significant bit to the most significant bit
  // This is done to reduce the number of matrix multiplications
  // by raising the input matrix in powers of 2
  // The total number of matrix multiplications are
  // number of bits + number of bits that equal 1 ~ O(log n)
  // instead of O(n)
  Tensor z, result;
  while (n > 0) {
    const auto bit = n % 2;
    n = n / 2;
    z = z.defined() ? at::matmul(z, z) : a;
    if (bit == 1) {
      if (_out.has_value() && n <= 0) {
        // Last multiplication can use the out version
        return result.defined() ? at::matmul_out(out, result, z) : out.copy_(z);
      }
      result = result.defined() ? at::matmul(result, z) : z;
    }
  }

  return result;
}

} // namespace

Tensor& linalg_matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  linalg_matrix_power_impl(self, n, result);
  return result;
}

Tensor linalg_matrix_power(const Tensor& self, int64_t n) {
  return linalg_matrix_power_impl(self, n, c10::nullopt);
}

Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return at::native::linalg_matrix_power_out(self, n, result);
}

Tensor matrix_power(const Tensor& self, int64_t n) {
  return at::native::linalg_matrix_power(self, n);
}

namespace {

// Computes the rank of 'input' and saves the result in-place in 'result'.
// 'hermitian' controls whether SVD or eigendecomposition is used for computing the singular values
// 'atol' and 'rtol' are the absolute and relative tolerances, respectively.
Tensor& matrix_rank_impl(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    bool hermitian,
    Tensor& result) {
  Tensor atol, rtol;
  std::tie(atol, rtol) = get_atol_rtol(input, atol_opt, rtol_opt, "torch.linalg.matrix_rank");

  checkSameDevice("torch.linalg.matrix_rank", result, input);
  checkSameDevice("torch.linalg.matrix_rank", atol, input, "atol");
  checkSameDevice("torch.linalg.matrix_rank", rtol, input, "rtol");
  ScalarType output_type = ScalarType::Long;
  checkLinalgCompatibleDtype("torch.linalg.matrix_rank", result.scalar_type(), output_type);

  checkNotComplexTolerance(atol, "torch.linalg.matrix_rank", "atol");
  checkNotComplexTolerance(rtol, "torch.linalg.matrix_rank", "rtol");

  // NumPy doesn't take into account possible input with no elements and it errors on max not defined for this case
  // Let's output 0 for this case, since that kind of matrices have zero number of non-zero rows, hence rank is 0.
  if (input.sym_numel() == 0) {
    result.fill_(0);
    return result;
  }

  // We compute matrix rank as the number of singular or absolute eigen values
  // that are above max(atol, rtol * max(S)) threshold
  Tensor S, max_S;
  if (!hermitian) {
    S = at::linalg_svdvals(input);
    // singular values are sorted in descending order
    max_S = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);
  } else {
    S = at::linalg_eigvalsh(input);
    S = S.abs();
    // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
    max_S = S.amax(/*dim=*/-1, /*keepdim=*/true);
  }

  Tensor tol = at::max(atol.unsqueeze(-1), rtol.unsqueeze(-1) * max_S);

  if (isTensorSubclassLike(input)) {
     result = at::sum(S > tol, /*dim=*/-1);
     return result;
  }

  result = at::sum_out(result, S > tol, /*dim=*/-1);
  return result;
}

Tensor get_matrix_rank_result_tensor(const Tensor& input) {
  // Matrices or batch of matrices are allowed
  checkIsMatrix(input, "torch.linalg.matrix_rank", "input");
  // For Composite Compliance, allocate `result` of correct shape to
  // avoid resizing in `out` variant.
  // See also `NOTE [matrix rank output shape]`
  auto result_shape =
      SymIntArrayRef(input.sym_sizes().cbegin(), input.sym_sizes().cend() - 2);
  Tensor result =
      at::empty_symint(result_shape, input.options().dtype(ScalarType::Long));

  return result;
}

}  // anonymous namespace

Tensor& linalg_matrix_rank_out(
    const Tensor& input,
    const optional<Tensor>& atol_opt,
    const optional<Tensor>& rtol_opt,
    bool hermitian,
    Tensor& result) {
  // Matrices or batch of matrices are allowed
  checkIsMatrix(input, "torch.linalg.matrix_rank", "input");
  auto result_shape =
    IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
  at::native::resize_output(result, result_shape);
  return matrix_rank_impl(input, atol_opt, rtol_opt, hermitian, result);
}

Tensor& linalg_matrix_rank_out(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian, Tensor& result) {
  Tensor atol_tensor, rtol_tensor;
  std::tie(atol_tensor, rtol_tensor) = get_atol_rtol(input, atol, rtol);
  result = linalg_matrix_rank_out(input, atol_tensor, rtol_tensor, hermitian, result);
  return result;
}

Tensor linalg_matrix_rank(const Tensor& input, const optional<Tensor>& atol, const optional<Tensor>& rtol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);
  return matrix_rank_impl(input, atol, rtol, hermitian, result);
}

Tensor linalg_matrix_rank(const Tensor& input, optional<double> atol, optional<double> rtol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);

  Tensor atol_tensor, rtol_tensor;
  std::tie(atol_tensor, rtol_tensor) = get_atol_rtol(input, atol, rtol);

  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}

Tensor& linalg_matrix_rank_out(const Tensor& input, const Tensor& tol, bool hermitian, Tensor& result) {
  // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
  // It is assumed that the provided value is the absolute tolerance
  Tensor rtol = at::zeros({}, tol.options());
  result = at::linalg_matrix_rank_outf(input, tol, rtol, hermitian, result);
  return result;
}

Tensor& linalg_matrix_rank_out(const Tensor& input, double tol, bool hermitian, Tensor& result) {
  // For NumPy compatibility tol is not scaled with max(singular_value) if the value for tol is provided
  // It is assumed that the provided value is the absolute tolerance
  result = at::linalg_matrix_rank_outf(input, tol, 0.0, hermitian, result);
  return result;
}

Tensor linalg_matrix_rank(const Tensor& input, const Tensor& tol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);
  return matrix_rank_impl(input, tol, at::zeros({}, tol.options()), hermitian, result);
}

Tensor linalg_matrix_rank(const Tensor& input, double tol, bool hermitian) {
  auto result = get_matrix_rank_result_tensor(input);

  Tensor atol_tensor, rtol_tensor;
  std::tie(atol_tensor, rtol_tensor) = get_atol_rtol(input, tol, 0.0);

  return matrix_rank_impl(input, atol_tensor, rtol_tensor, hermitian, result);
}

// multi_dot helper functions
namespace {

/**
 * @brief Computes the optimal matrix chain multiplication order
 *
 * Follows the dynamic programming algorithm from Cormen et al,
 * "Introduction to Algorithms, Third Edition", Chapter 15.2,
 * p. 370-378. Note that the book uses 1-based indexing.
 *
 * The cost of multiplying two matrices with sizes p x q and q x r
 * is defined here as p * q * r. The optimal multiplication order
 * is the one that minimizes the total cost.
 *
 * @param tensors list of 2D tensors
 * @return a 2D vector s used by #matrix_chain_multiplication to construct
 *         the optimal matrix multiplication order. The optimal multiplication
 *         order for multiplying tensors i...j is to multiply tensors i...s[i, j]
 *         and tensors (s[i, j] + 1)...j first and then the result of that.
 */
std::vector<std::vector<int64_t>> matrix_chain_order(TensorList tensors) {
  const size_t n = tensors.size();

  // Tensor i has dimensions p[i] x p[i + 1]
  std::vector<int64_t> p(n + 1);
  for (const auto i : c10::irange(n)) {
    p[i] = tensors[i].size(0);
  }
  p[n] = tensors[n - 1].size(1);

  // m[i, j] = k where k is the minimum cost for multiplying tensors i...j
  std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

  // s[i, j] = k where k is the index at which to split the list such that
  // optimally multiplying matrices i...k and k...j first and then the resulting
  // matrices is the optimal order for multiplying matrices i...j.
  std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

  // Compute the optimal multiplication order
  for (const auto l : c10::irange(1, n)) {
    for (const auto i : c10::irange(n - l)) {
      const auto j = i + l;
      m[i][j] = std::numeric_limits<int64_t>::max();
      for (const auto k : c10::irange(i, j)) {
        const auto q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
        if (q < m[i][j]) {
          m[i][j] = q;
          s[i][j] = k;
        }
      }
    }
  }

  return s;
}

/**
 * @brief Recursively multiplies the tensors i...j using the given order
 *
 * @param tensors matrices to multiply together
 * @param order optimal chain multiplication order from #matrix_chain_order
 * @param i index of first tensor to be multiplied
 * @param j index of last tensor to be multiplied
 * @return Tensor result of multiplying tensors[i...j] together.
 */
Tensor matrix_chain_multiplication(
    TensorList tensors,
    const std::vector<std::vector<int64_t>>& order,
    int64_t i,
    int64_t j) {
  if (i == j) {
    return tensors[i];
  }
  return at::mm(
      matrix_chain_multiplication(tensors, order, i, order[i][j]),
      matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
}

// Implements torch.linalg.multi_dot
Tensor multi_dot_impl(TensorList _tensors, c10::optional<Tensor> _out) {
  const size_t n = _tensors.size();
  TORCH_CHECK(n >= 2, "multi_dot(): expected at least 2 tensors but got ", n);

  std::vector<int64_t> out_shape;
  std::vector<Tensor> tensors(n);

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (_tensors[0].dim() == 1) {
    tensors[0] = _tensors[0].unsqueeze(0);
  } else if (_tensors[0].dim() == 2) {
    tensors[0] = _tensors[0];
    out_shape.emplace_back(tensors[0].size(0));
  } else {
    TORCH_CHECK(
        false,
        "multi_dot(): the first tensor must be 1D or 2D but got ",
        _tensors[0].dim(),
        "D");
  }

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (_tensors[n - 1].dim() == 1) {
    tensors[n - 1] = _tensors[n - 1].unsqueeze(-1);
  } else if (_tensors[n - 1].dim() == 2) {
    tensors[n - 1] = _tensors[n - 1];
    out_shape.emplace_back(tensors[n - 1].size(1));
  } else {
    TORCH_CHECK(
        false,
        "multi_dot(): the last tensor must be 1D or 2D but got ",
        _tensors[n - 1].dim(),
        "D");
  }

  // Ensure middle tensors are 2D
  for (const auto i : c10::irange(1, n - 1)) {
    TORCH_CHECK(
        _tensors[i].dim() == 2,
        "multi_dot(): tensor ",
        i,
        " must be 2D but got ",
        _tensors[i].dim(),
        "D");
    tensors[i] = _tensors[i];
  }

  // Ensure all tensors have the same device and dtype and check
  // that the shapes can be multiplied
  const auto dtype = tensors[0].dtype();
  const auto device = tensors[0].device();
  for (const auto i : c10::irange(1, n)) {
    TORCH_CHECK(
        tensors[i].dtype() == dtype,
        "multi_dot(): all tensors must have be the same dtype but tensor 0 is ",
        dtype,
        " and tensor ",
        i,
        " ",
        tensors[i].dtype());
    TORCH_CHECK(
        tensors[i].device() == device,
        "multi_dot(): all tensors must be on the same device but tensor 0 is on ",
        device,
        " and tensor ",
        i,
        " on ",
        tensors[i].device());
    TORCH_CHECK(
        tensors[i - 1].size(-1) == tensors[i].size(0),
        "multi_dot(): tensors ",
        i - 1,
        " and ",
        i,
        " with shapes ",
        _tensors[i - 1].sizes(),
        " and ",
        _tensors[i].sizes(),
        " cannot be multiplied")
  }

  Tensor result;

  if (_out.has_value()) {
    auto out = *_out;
    TORCH_CHECK(
        dtype == out.dtype(),
        "multi_dot(): expected out tensor to have dtype ",
        dtype,
        " but got ",
        out.dtype());
    TORCH_CHECK(
        device == out.device(),
        "multi_dot(): expected out tensor to be on device ",
        device,
        " but got ",
        out.device());

    // If the last and last tensors have shapes (a, b) and (b, c) the
    // output has shape (a, c). If either the first or last tensor is 1D
    // a and/or c dimensions will be implicitely size 1 and will be ommited
    // from the output. e.g. for inputs (a, b) x (b) the output has shape (a,).
    at::native::resize_output(out, out_shape);

    // View output as 2D for simplicity of computation.
    result = out.view({tensors[0].size(0), tensors.back().size(-1)});
  }

  // The resize_ and view calls below are to ensure the
  // output shape respects the original dimensionality of
  // the first and last tensors which we are now viewed as 2D

  if (tensors.size() == 2) {
    return _out.has_value() ? at::mm_out(result, tensors[0], tensors[1])
                         : at::mm(tensors[0], tensors[1]).view(out_shape);
  }

  // Why the separate implementation for 3 matrices?
  // The logic for three matrices is much faster when done directly
  // Requires 1 comparison to 4 comparisons and fewer arithmetic operations
  if (tensors.size() == 3) {
    const auto a = tensors[0].size(0);
    const auto b = tensors[1].size(0);
    const auto c = tensors[2].size(0);
    const auto d = tensors[2].size(1);

    // The matrices are of size (a x b), (b x c), (c x d)
    // cost_1 is the cost of parenthesizing (a x b) and (b x c) and then
    // combining (c x d) cost_2 is the cost of parenthesizing (b x c) and (c x
    // d) and then combining (a x b)
    const auto cost_1 = (a * c) * (b + d);
    const auto cost_2 = (b * d) * (a + c);

    if (cost_1 > cost_2) {
      return _out.has_value()
          ? at::mm_out(result, tensors[0], at::mm(tensors[1], tensors[2]))
          : at::mm(tensors[0], at::mm(tensors[1], tensors[2])).view(out_shape);
    } else {
      return _out.has_value()
          ? at::mm_out(result, at::mm(tensors[0], tensors[1]), tensors[2])
          : at::mm(at::mm(tensors[0], tensors[1]), tensors[2]).view(out_shape);
    }
  }

  // Algorithm for multiplying 4 or more matrices
  const auto order = matrix_chain_order(tensors);
  const int64_t i = 0;
  const int64_t j = n - 1;

  if (_out.has_value()) {
    // We manually implement the first recursive layer here so we can use mm_out
    // for the final multiplication
    return at::mm_out(
        result,
        matrix_chain_multiplication(tensors, order, i, order[i][j]),
        matrix_chain_multiplication(tensors, order, order[i][j] + 1, j));
  }
  return matrix_chain_multiplication(tensors, order, i, j).view(out_shape);
}

} // namespace

Tensor linalg_multi_dot(TensorList tensors) {
  return multi_dot_impl(tensors, c10::nullopt);
}

Tensor& linalg_multi_dot_out(TensorList tensors, Tensor& result) {
  multi_dot_impl(tensors, result);
  return result;
}

Tensor chain_matmul(TensorList matrices) {
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  checkAllSameDim(matrices, 2);

  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  if (matrices.size() == 1) {
    return matrices[0].clone();
  }

  return at::native::linalg_multi_dot(matrices);
}

Tensor& chain_matmul_out(TensorList matrices, Tensor& result) {
  TORCH_WARN_ONCE(
      "torch.chain_matmul is deprecated and will be removed in a future PyTorch release. ",
      "Use torch.linalg.multi_dot instead, which accepts a list of two or more tensors rather than ",
      "multiple parameters."
  );
  checkAllSameDim(matrices, 2);

  TORCH_CHECK(
      !matrices.empty(), "chain_matmul(): Expected one or more matrices");

  if (matrices.size() == 1) {
    at::native::resize_output(result, matrices[0].sizes());
    return result.copy_(matrices[0]);
  }

  return at::native::linalg_multi_dot_out(matrices, result);
}

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
 TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

static void check_addr_scalar(const ScalarType dtype,
                              const Scalar& scalar,
                              const std::string& scalar_name) {
  TORCH_CHECK(
    !scalar.isBoolean() || dtype == ScalarType::Bool,
    "Boolean ", scalar_name, " only supported for Boolean results.");
  TORCH_CHECK(
    isFloatingType(dtype) || isComplexType(dtype) || scalar.isIntegral(true),
    "For integral input tensors, "
    "argument ", scalar_name ," must not be a floating point number.");
}

static TensorIterator build_addr_iter(Tensor& result,
                                      const Tensor& self,
                                      const Tensor& vec1,
                                      const Tensor& vec2) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  const auto vec1_size0 = vec1.sizes()[0];
  const auto vec2_size0 = vec2.sizes()[0];
  auto self_ = &result == &self
    ? c10::MaybeOwned<Tensor>::borrowed(self)
    : expand_size(self, {vec1_size0, vec2_size0}, "addr");
  TORCH_CHECK(
    self_->dim() == 2,
    "2D tensor expected, got ", self_->dim(), "D tensor for input"
  );
  TORCH_CHECK(
    self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
    "size mismatch, input: ", self_->sizes(),
    ", v1: ", vec1.sizes(),
    ", v2: ", vec2.sizes()
  );

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .add_output(result)
    .add_owned_input(*self_)
    .add_owned_input(vec1.reshape({vec1_size0, 1}))
    .add_input(vec2)
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
  return iter;
}

Tensor addr(const Tensor& self,
            const Tensor& vec1, const Tensor& vec2,
            const Scalar& beta, const Scalar& alpha) {
  Tensor result;
  auto iter = build_addr_iter(result, self, vec1, vec2);

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  addr_stub(iter.device_type(), iter, beta, alpha);
  return iter.output();
}

Tensor& addr_(Tensor& self,
              const Tensor& vec1, const Tensor& vec2,
              const Scalar& beta, const Scalar& alpha) {
  return at::addr_out(self, self, vec1, vec2, beta, alpha);
}

Tensor& addr_out(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha, Tensor &result) {
  auto iter = build_addr_iter(result, self, vec1, vec2);

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  addr_stub(iter.device_type(), iter, beta, alpha);
  return result;
}

// The math_addr and math_addr_out functions support backends
// other than CPU and CUDA, such as XLA.
// They are implemented using the composition of existing ops
Tensor math_addr(const Tensor& self,
                 const Tensor& vec1, const Tensor& vec2,
                 const Scalar& beta, const Scalar& alpha) {
  // when beta==0, values in self should be ignored,
  // nans and infs in self should not propagate.
  Tensor out;
  if (beta.toComplexDouble() == 0.0) {
    if (alpha.toComplexDouble() == 1.0) {
      out = at::outer(vec1, vec2);
    } else {
      out = alpha * at::outer(vec1, vec2);
    }
  } else if (beta.toComplexDouble() == 1.0) {
    if (alpha.toComplexDouble() == 1.0) {
      out = self + at::outer(vec1, vec2);
    } else {
      out = self + alpha * at::outer(vec1, vec2);
    }
  } else if (alpha.toComplexDouble() == 1.0) {
    out = beta * self + at::outer(vec1, vec2);
  } else {
    out = beta * self + alpha * at::outer(vec1, vec2);
  }
  auto result_type = c10::promoteTypes(c10::promoteTypes(self.scalar_type(), vec1.scalar_type()), vec2.scalar_type());
  return out.to(c10::TensorOptions().dtype(result_type));
}

Tensor& math_addr_out(const Tensor& self,
                      const Tensor& vec1, const Tensor& vec2,
                      const Scalar& beta, const Scalar& alpha, Tensor &result) {
  auto addr_result = at::addr(self, vec1, vec2, beta, alpha);

  // Validates safe casting
  const auto result_dtype = addr_result.scalar_type();
  TORCH_CHECK(canCast(result_dtype, result.scalar_type()),
              "result type ", result_dtype,
              " can't be cast to the desired output type ", result.scalar_type());

  at::native::resize_output(result, addr_result.sizes().vec());
  result.copy_(addr_result);
  return result;
}

// torch.ger, alias for torch.outer
Tensor& ger_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  TORCH_WARN("torch.ger is deprecated and will be removed in a future PyTorch release. "
             "Use torch.outer instead.");
  return at::outer_out(result, self, vec2);
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  return self.outer(vec2);
}

Tensor& inner_out(const Tensor& self, const Tensor& other, Tensor& out) {
  checkDeviceType("inner()", {out, self, other}, self.device().type());

  // If either self or other is a scalar just multiply them
  if (self.dim() == 0 || other.dim() == 0) {
    at::mul_out(out, self, other);
    return out;
  }

  // Last dimension should match (tensordot does not enforce this)
  TORCH_CHECK(
      self.size(-1) == other.size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sizes(),
      " and ",
      other.sizes());

  at::tensordot_out(out, self, other, -1, -1);
  return out;
}

Tensor inner(const Tensor& self, const Tensor& other) {
  checkDeviceType("inner()", {self, other}, self.device().type());

  // If either self or other is a scalar just multiply them
  if (self.dim() == 0 || other.dim() == 0) {
    return self * other;
  }

  // Last dimension should match (tensordot does not enforce this)
  TORCH_CHECK(
      self.sym_size(-1) == other.sym_size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sym_sizes(),
      " and ",
      other.sym_sizes());

  return at::tensordot(self, other, -1, -1);
}

Tensor& outer_out(const Tensor& self, const Tensor& vec2, Tensor &result) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  // torch.outer is implemented as a composite op using reshape and mul
  at::mul_out(result, self.reshape({self.size(0), 1}), vec2);
  return result;
}

Tensor outer(const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  return self.reshape_symint({self.sym_size(0), 1}) * vec2;
}

static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

  TORCH_CHECK(
    m1.dtype() == m2.dtype(),
    "expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype()
  )
  // Array access is faster than .size(n) and .stride(n)
  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");

  at::native::resize_output(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return;
  }

  // Some paths in the code below do not handle multiplications of the form [a, 0] x [0, b]
  if (m1_sizes[1] == 0) {
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      if (!self.is_same(result)) {
        result.copy_(self);
      }
      result.mul_(beta);
    }
    return;
  }

  if (beta.toComplexDouble() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  bool transpose_c = false;
  Tensor c;

  // Cast result as matrix a
  if (result_strides[0] == 1 &&
      (result_sizes[1] == 1 || result_strides[1] >= std::max(int64_t{1}, result_sizes[0]))) {
    transpose_c = false;
    c = result.resolve_conj();
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result.resolve_conj();
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.resolve_conj().transpose(0, 1).contiguous().transpose_(0, 1);
  }

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1_sizes[transpose_c ? 0 : 1];

  // Cast m1 as matrix a
  bool transpose_a = false;
  Tensor a;
  /* Need lda >= max(1, (transpose_a ? k : m)) */
  if (m1_strides[transpose_c ? 1 : 0] == 1 &&
      m1_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, m)) {
    transpose_a = false;
    a = m1.resolve_conj();
  } else if (m1_strides[transpose_c ? 0 : 1] == 1 &&
             m1_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, k)) {
    transpose_a = true;
    a = m1;
  } else {
    transpose_a = !transpose_c;
    a = m1.clone(at::MemoryFormat::Contiguous);
  }

  // Cast m2 as matrix b
  bool transpose_b = false;
  Tensor b;
  /* Need ldm2_ >= max(1, (transpose_m2 == 'n' ? k : n)) */
  if (m2_strides[transpose_c ? 1 : 0] == 1 &&
      m2_strides[transpose_c ? 0 : 1] >= std::max(int64_t{1}, k)) {
    transpose_b = false;
    b = m2.resolve_conj();
  } else if (m2_strides[transpose_c ? 0 : 1] == 1 &&
             m2_strides[transpose_c ? 1 : 0] >= std::max(int64_t{1}, n)) {
    transpose_b = true;
    b = m2;
  } else {
    transpose_b = !transpose_c;
    b = m2.clone(at::MemoryFormat::Contiguous);
  }

  const int64_t lda = a.strides()[(transpose_a == transpose_c) ? 1 : 0];
  const int64_t ldb = b.strides()[(transpose_b == transpose_c) ? 1 : 0];
  const int64_t ldc = c.strides()[transpose_c ? 0 : 1];

  // Always ensure the conjugation for c is resolved since there's no way to specify c's conjugation in the gemm call
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c.is_conj());

  bool dispatched = false;
#if defined(__aarch64__) && AT_MKLDNN_ACL_ENABLED()
  // On AArch64 if LHS matrix in BLAS routine is transposed but RHS is not then
  // it is faster to call oneDNN matrix multiplication primitive with RHS*LHS
  // that will call then into Arm® Compute Library (ACL) GEMM kernel and also
  // additionally have support for running kernel with BF16 instructions
  if(transpose_a && !transpose_b && result.scalar_type() == at::ScalarType::Float) {
      mkldnn_matmul(b, a, c, beta.to<float>(), alpha.to<float>());
      // We have dispatched to ACL GEMM for single precision float
      // so do not need to dispatch to BLAS GEMM below
      dispatched = true;
  }
#endif

  if(!dispatched) {
    // Apply BLAS routine
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16,
        result.scalar_type(), "addmm_impl_cpu_",
        [&]{
          using opmath_t = at::opmath_type<scalar_t>;
          at::native::cpublas::gemm(
              transpose_a ? a.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
              transpose_b ? b.is_conj() ? TransposeType::ConjTranspose : TransposeType::Transpose : TransposeType::NoTranspose,
              m, n, k,
              alpha.to<opmath_t>(),
              a.const_data_ptr<scalar_t>(), lda,
              b.const_data_ptr<scalar_t>(), ldb,
              beta.to<opmath_t>(),
              c.mutable_data_ptr<scalar_t>(), ldc);
        });
  }

  if (!c.is_same(result)) {
    result.copy_(c);
  }
}

static void addbmm_impl_(
    Tensor &result, const Tensor &self, const Tensor &batch1, const Tensor &batch2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  TORCH_CHECK(batch1.size(0) == batch2.size(0),
      "batch1 and batch2 must have same number of batches, got ",
      batch1.size(0), " and ", batch2.size(0));
  TORCH_CHECK(batch1.size(2) == batch2.size(1),
      "Incompatible matrix sizes for bmm (",
      batch1.size(1), "x", batch1.size(2), " and ",
      batch2.size(1), "x", batch2.size(2), ")");

  const int64_t dim1 = batch1.size(1);
  const int64_t dim2 = batch2.size(2);
  TORCH_CHECK(self.size(0) == dim1 && self.size(1) == dim2,
      "self tensor does not match matmul output shape");

  result.resize_as_(self);

  if (beta.to<c10::complex<double>>() != 0.0 && !self.is_same(result)) {
    result.copy_(self);
  }

  const int64_t num_batches = batch1.size(0);

  if (num_batches == 0) {
    if (beta.to<c10::complex<double>>() != 0.0) {
      result.mul_(beta);
    } else {
      result.zero_();
    }
    return;
  }

  auto adjusted_beta(beta);
  for (const auto batch : c10::irange(num_batches)) {
    result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);
    adjusted_beta = 1; // accumulate output once
  }
}

Tensor& addbmm_out(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor& result) {
  auto b_self = expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out");
  {
    at::NoNamesGuard guard;
    addbmm_impl_(result, *b_self, batch1, batch2, beta, alpha);
  }
  auto names = at::namedinference::propagate_names_for_addmm(batch1, batch2, self);
  at::namedinference::propagate_names_if_nonempty(result, names);
  return result;
}

Tensor &addbmm_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return native::addbmm_out(self, batch1, batch2, beta, alpha, self);
}

Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return native::addbmm_out(self, batch1, batch2, beta, alpha, result);
}

TORCH_IMPL_FUNC(addmm_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, const Tensor &result) {
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
  }
}

TORCH_IMPL_FUNC(addmm_activation_out_cpu)(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha, bool use_gelu, const Tensor &result) {
  auto b_self = expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out");
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), *b_self, mat1, mat2, beta, alpha);
    if (use_gelu) {
      at::gelu_(const_cast<Tensor&>(result));
    } else {
      at::relu_(const_cast<Tensor&>(result));
    }
  }
}

TORCH_IMPL_FUNC(mm_out_cpu)(const Tensor & self, const Tensor & mat2, const Tensor & result) {
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(const_cast<Tensor&>(result), result, self, mat2, 0, 1);
  }
}

template <typename scalar_t, bool is_bmm>
inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  int64_t bs = result.size(0);
  int64_t is = result.size(1);
  int64_t js = result.size(2);
  int64_t ks = self.size(2);

  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t alpha = alpha_.to<opmath_t>();
  opmath_t beta = beta_.to<opmath_t>();

  auto r0 = result.accessor<scalar_t, 3>();
  auto s0 = self.accessor<scalar_t, 3>();
  auto m0 = mat2.accessor<scalar_t, 3>();

  int64_t grain_size = std::max(internal::GRAIN_SIZE / (is * js * ks), (int64_t)1);
  using opmath_t = at::opmath_type<scalar_t>;
  parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
      for (const auto b : c10::irange(b_begin, b_end)) {
        auto r1 = r0[b];
        auto s1 = s0[b];
        auto m1 = m0[b];
        for (const auto i : c10::irange(is)) {
          auto r2 = r1[i];
          auto s2 = s1[i];
          for (const auto j : c10::irange(js)) {
            opmath_t acc_value = 0;//is_bmm ? opmath_t(0) : opmath_t(r2[j]);
            for (const auto k : c10::irange(ks)) {
              acc_value += static_cast<opmath_t>(s2[k]) *
                  static_cast<opmath_t>(m1[k][j]);
            }
            if (is_bmm) {
              r2[j] = acc_value;
            } else {
              // For beta == 0, the r's value will be ignored, especially for nan value.
              if (beta == opmath_t{0}) {
                r2[j] = alpha * acc_value;
              } else {
                r2[j] = static_cast<opmath_t>(r2[j]) * beta + alpha * acc_value;
              }
            }
          }
        }
      }
    });
}

void baddbmm_with_gemm_(const Tensor &result, const Tensor &mat1, const Tensor &mat2, const Scalar &beta_, const Scalar &alpha_) {
  TORCH_INTERNAL_ASSERT(result.is_contiguous());

  const auto result_sizes = result.sizes();
  const auto result_strides = result.strides();
  const auto mat1_strides = mat1.strides();
  const auto mat2_strides = mat2.strides();
  const auto mat1_sizes = mat1.sizes();
  const auto mat2_sizes = mat2.sizes();

  auto is_transposed = [](const c10::IntArrayRef& strides, const c10::IntArrayRef& sizes) {
    return strides[1] == 1 && strides[2] >= sizes[1];
  };

  // gemm expects fortran order matrices, so we swap argument order to transpose everything
  const auto transpose_a = is_transposed(mat2_strides, mat2_sizes);
  const auto transpose_b = is_transposed(mat1_strides, mat1_sizes);

  const int64_t batch_size = mat1_sizes[0];
  const int64_t m = result_sizes[2];
  const int64_t n = result_sizes[1];
  const int64_t k = mat2_sizes[1];

  const int64_t lda = mat2_strides[transpose_a ? 2 : 1];
  const int64_t ldb = mat1_strides[transpose_b ? 2 : 1];
  const int64_t ldc = result_strides[1];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "baddbmm_with_gemm", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto alpha = alpha_.to<opmath_t>();
    const auto beta = beta_.to<opmath_t>();
    at::native::cpublas::gemm_batched_with_stride(
        transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
        transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
        batch_size, m, n, k, alpha,
        mat2.data_ptr<scalar_t>(), lda, mat2_strides[0],
        mat1.data_ptr<scalar_t>(), ldb, mat1_strides[0],
        beta,
        result.data_ptr<scalar_t>(), ldc, result_strides[0]);
  });
}

// This tries to apply some optimizations to bmm/baddbmm:
// - When the operand size is small, computation are parallelized over the batch
//   dimension using OMP and naive matrix multiplication is applied.
// - When the operand size is larger than the threshold, if compiled with MKL, MKL's batch gemm is used.
// - Otherwise, we use a series of matrix multiplications.
// The threshold of 400 for the first has not been thoroughly benchmarked yet and may have room for further
// optimization, it likely depends on the characteristics of the CPU, MKL will be different from non-MKL etc.,
// but this seems to be a first starting point.

static inline void bmm_out_or_baddbmm_(const Tensor& self_or_result_, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm_out) {
  // is_bmm_out: true for bmm_out, false for baddbmm_
  // self_or_result is "self" for baddbmm_ and "result" for bmm_out
  Tensor& self_or_result = const_cast<Tensor&>(self_or_result_);

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];

  // handle pathological cases that blas may not like
  if (self_or_result.numel() == 0) {
    return;
  } else if (contraction_size == 0) {
    if (is_bmm_out || (beta.to<c10::complex<double>>() == 0.0)) {
      self_or_result.zero_();
      return;
    } else {
      self_or_result.mul_(beta);
      return;
    }
  }

  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    return (strides[2] == 1 && strides[1] >= sizes[2])
            || (strides[1] == 1 && strides[2] >= sizes[1]);
  };

  if (use_mkldnn_bf16_matmul(batch1, batch2, self_or_result)){
    mkldnn_matmul(batch1, batch2, self_or_result, beta.to<float>(), alpha.to<float>());
    return;
  }

  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, batch1.scalar_type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, batch1.scalar_type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else if (at::hasMKL() && ((
            self_or_result.scalar_type() != kBFloat16 &&
            at::native::is_floating_point(self_or_result)) ||
            at::native::is_complex(self_or_result))
            && batch_items_contiguous_or_transposed(batch1)
            && batch_items_contiguous_or_transposed(batch2)
            && self_or_result.is_contiguous()) {
    baddbmm_with_gemm_(self_or_result, batch1, batch2, beta, alpha);
  } else { // split along batch dimension
#ifdef C10_MOBILE
    /*
     * We only do multithreading when Inference mode is enabled because various
     * thread local state is not appropriately propagated through
     * at::parallel_for. e.g. RecordFunction related state, dispatchKeySet Big
     * concern with this is that if we use at::parallel_for where state is not
     * propagated then dispatch machinery may work differently on main thread
     * vs. other threads, leading to undefined behavior.
     * Thus it is recommended to not use at::parallel_for where lambdas do
     * ops that go through dispatcher.
     * For now we circument this by InferenceMode guard in order to unlock
     * performance.
     * Longer term we probably want a separate API that explicitly calls out
     * the TLS that it propagates.
     * Also note that this is enabled for mobile only because blas
     * implementation for non-mobile build is already multithreaded.
     */
    // Benchmarking was done as follows:
    // bmm_test: operator benchmark under
    // benchmarks/operator_benchmarks/pt/bmm_test.py Ran this benchmark for
    // various matrix sizes on Samsung S8U
    const bool enable_multithreaded_bmm = c10::InferenceMode::is_enabled() &&
        bs >= 4 && res_rows >= 4 && res_cols >= 16 && contraction_size >= 16;
#else
    const bool enable_multithreaded_bmm{false};
#endif
    if (is_bmm_out) {
      if (enable_multithreaded_bmm) {
        auto bmm_out_fn = [&](uint64_t start, uint64_t end) {
          c10::InferenceMode guard;
          for (const auto b : c10::irange(start, end)) {
            auto r = self_or_result.select(0, b);
            addmm_impl_cpu_(
                r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
          }
        };
        at::parallel_for(0, bs, 1, bmm_out_fn);
      } else {
        for (const auto b : c10::irange(bs)) {
          auto r = self_or_result.select(0, b);
          addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
        }
      }
    } else {
      if (enable_multithreaded_bmm) {
        auto bmm_fn = [&](uint64_t start, uint64_t end) {
          c10::InferenceMode guard;
          for (const auto b : c10::irange(start, end)) {
            self_or_result.select(0, b).addmm_(
                batch1.select(0, b), batch2.select(0, b), beta, alpha);
          }
        };
        at::parallel_for(0, bs, 1, bmm_fn);
      } else {
        for (const auto b : c10::irange(bs)) {
          self_or_result.select(0, b).addmm_(
              batch1.select(0, b), batch2.select(0, b), beta, alpha);
        }
      }
    }
  }
  return;
}

void conjugate_mutable_input_if_needed(const Tensor& self, bool conjugate) {
  if (conjugate) {
    self.conj_physical_();
  }
}

TORCH_IMPL_FUNC(baddbmm_out_cpu)
(const Tensor & self, const Tensor & batch1, const Tensor & batch2, const Scalar& beta, const Scalar& alpha, const Tensor& result) {
    bool self_is_conj = result.is_conj();
    conjugate_mutable_input_if_needed(result, self_is_conj);
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), beta, alpha, false);
    conjugate_mutable_input_if_needed(result, self_is_conj);
  }

TORCH_IMPL_FUNC(bmm_out_cpu)
(const Tensor & batch1, const Tensor & batch2, const Tensor & result) {
    {
    NoNamesGuard guard;
    bool result_is_conj = result.is_conj();
    conjugate_mutable_input_if_needed(result, result_is_conj);
    bmm_out_or_baddbmm_(result, batch1.resolve_conj(), batch2.resolve_conj(), Scalar(0.0), Scalar(1.0), true);
    conjugate_mutable_input_if_needed(result, result_is_conj);
    }
}

Tensor& dot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "dot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  return result.fill_(self.dot(other));
}

Tensor& vdot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto output_device = result.device();
  auto input1_device = self.device();
  auto input2_device = other.device();
  // check if the input & output tensors are on the same device.
  TORCH_CHECK(
    (output_device == input1_device) && (input1_device == input2_device),
    "vdot: Expected the output and input tensors to be on the "
    "same device, but got the output tensor on ", output_device,
    ", the 'input' tensor on ", input1_device, ", and the 'other' tensor on ", input2_device);
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match input dtype ", self.scalar_type());
  return result.fill_(self.vdot(other));
}

bool should_fold(const Tensor& tensor1, const Tensor& tensor2) {
  // We check that we can fold the larger tensor into a matrix and dispatch to mm or mv rather than
  // to bmm. We want to make sure we can do so without incurring in any extra copy
  const auto tensor1_larger = tensor1.dim() >= tensor2.dim();

  // We order the tensors. t1 will be the larger tensor
  // We can always transpose tensor2 as the dimensions are always >= 1 (precondition from matmul)
  // and tensor1_larger iff tensor2.dim() > tensor1.dim(9
  const auto t1 = tensor1_larger ? MaybeOwned<Tensor>::borrowed(tensor1)
                                 : MaybeOwned<Tensor>::owned(tensor2.mT());
  const int64_t dim_t1 = t1->dim();
  const auto dim_t2 = tensor1_larger ? tensor2.dim()
                                     : tensor1.dim();

  // Just fold for dim_t1 >= 3 and (dim_t2 == 1 || dim_t2 == 2)
  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }

  // In this case we *do* incur in an extra copy to avoid creating an unnecessary large tensor in the backward
  // Suppose we don't fold here. Let t1.shape = [b, m, n] t2.shape = [n, k] like in a transformer
  // t2 will be expanded to a tensor of shape [b, n, k] and then we do t1.bmm(t2_expanded)
  // The issue appears in the backward.
  // The output gradient g of this operation would have shape [b, m, k]
  // The backward wrt. t2 of bmm would be given by t1.mH @ g, which has shape [b, n, k]
  // Then, the backward of expand is simply `sum(0)`. As such, we are instantiating a tensor
  // of shape [b, n, k] unnacessarily, which may cause a large memory footprint, and in the
  // worst case, an OOM
  bool t2_requires_grad = tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad) {
    return true;
  }

  // Don't fold in this case, as we would have to call mm on the transposed tensor, the result
  // would be contiguous, and then we would need to transpose it and call contiguous on it, thus
  // having to copy the tensor
  if (tensor1.dim() == 2) {
    return false;
  }

  // Can always fold if the tensor is empty
  // This serves as a precondition for the code below
  if (t1->numel() == 0) {
    return true;
  }

  // t1->view(-1, t1->size(-1)) does not copy only when the first n-1 dimensions are contiguous
  // in the sense that t1_stride[i] = t1_stride[i+1]*t1_shape[i+1]
  const auto t1_shape = t1->sizes();
  const auto t1_strides = t1->strides();
  for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
    if (t1_strides[i] != t1_strides[i+1] * t1_shape[i+1]) {
      return false;
    }
  }
  return true;
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, (1d) the dot product (scalar) is returned.
- If the arguments are 2D - 1D or 1D - 2D, the matrix-vector product is returned.
- If both arguments are 2D, the matrix-matrix product is returned.
- If one of the arguments is ND with N >= 3 and the other is 1D or 2D, and some
  conditions on the strides apply (see should_fold) we fold the first N-1 dimensions
  of the ND argument to form a matrix, call mm or mv, reshape it back to ND and return it
- Otherwise, we return bmm, after broadcasting and folding the batched dimensions if
  there's more than one
*/
Tensor _matmul_impl(
    Tensor& out,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  NoNamesGuard guard;
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we just evaluate
  // the condition and move on
  TORCH_CHECK(dim_tensor1 != 0 && dim_tensor2 != 0,
              "both arguments to matmul need to be at least 1D, but they are ",
              dim_tensor1, "D and ", dim_tensor2, "D");


  const bool has_out = out.defined();

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (should_fold(tensor1, tensor2)) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // and at least one of the following two conditions hold
    // - the small tensor requires grad (see should_fold for the why)
    // - we can fold the larger tensor t1 into a matrix as t1.view(-1, t1.size(-1)) without copying

    // optimization: use mm instead of bmm by folding the batch of the larger tensor
    // into its leading matrix dimension
    const auto transpose = dim_tensor2 > dim_tensor1;
    const auto t1 = transpose ? MaybeOwned<Tensor>::owned(tensor2.mT())
                              : MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2 = !transpose ? MaybeOwned<Tensor>::borrowed(tensor2)
                               : dim_tensor1 == 2
                                   ? MaybeOwned<Tensor>::owned(tensor1.t())
                                   : MaybeOwned<Tensor>::borrowed(tensor1);
    // Invariant: t1->dim() >= 3 && (t2->dim() == 1 || t2->dim() == 2)
    //            and *t1 and *t2 are matmul-compatible

    // Why not t1->view(-1, sizes_1.back())?
    // If the last dim is 0, then view(-1, 0) won't work because the -1 becomes ambiguous.
    // This can happen in e.g. [3, 5, 0] @ [0, 0].
    const auto sizes_1 = t1->sizes();
    auto output_shape = DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);

    // Readjust output_shape if we are multiplying by a matrix
    const auto t2_is_matrix = t2->dim() == 2;
    if (t2_is_matrix) {
      output_shape.push_back(t2->sizes()[1]);
    }
    // This will almost always be a view.
    // It may not be a view if t2->requires_grad(). See should_fold for an explanation
    const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});
    if (!has_out) {
      if (t2_is_matrix) {
        const auto output = at::_unsafe_view(t1_folded.mm(*t2), output_shape);
        // This copies if we perform a 2D @ 3D and the first tensor requires_grad
        // See should_fold for why.
        // If mm_out were differentiable, we could use it here, and pass a result with the
        // correct strides to avoid this unnecessary copy.
        return transpose ? output.mT().contiguous() : output;
      } else {
        return at::_unsafe_view(t1_folded.mv(*t2), output_shape);
      }
    } else {
      // See the !has_out branch for an explanation
      TORCH_INTERNAL_ASSERT(!(transpose && t2_is_matrix));

      // Resize output into the correct shape
      at::native::resize_output(out, output_shape);

      // We then reshape the output to the expected shape and call mm/mv
      // and transpose back if necessary
      auto reshaped_out = t2_is_matrix ? out.reshape({folded_dim1, t2->sizes().back()})
                                       : out.reshape({folded_dim1});
      if (t2_is_matrix) {
        at::mm_out(reshaped_out, t1_folded, *t2);
      } else {
        at::mv_out(reshaped_out, t1_folded, *t2);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out);
      }
      return out;
    }
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // We track m1 vs m2 separately even though they must match for nicer error messages
    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    auto batch_tensor1 = tensor1.sizes().slice(0, std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 = dim_tensor2 > 1 ? tensor2.sizes().cend()[-2] : tensor2.sizes().front();
    const int64_t p = dim_tensor2 > 1 ? tensor2.sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(tensor2.sizes().data(),
                                    std::max<int64_t>(dim_tensor2 - 2, 0LL));

    // Same optimization for the gradients as that in should_fold
    // If we're going to broadcast we force it to go through the should_fold branch
    if (dim_tensor1 == 3 && dim_tensor2 == 3 && batch_tensor1[0] != batch_tensor2[0]) {
      if (batch_tensor1[0] == 1 && (tensor1.requires_grad() || isTensorSubclassLike(tensor1))) {
        return _matmul_impl(out, tensor1.squeeze(0), tensor2);
      }
      if (batch_tensor2[0] == 1 && (tensor2.requires_grad() || isTensorSubclassLike(tensor2))) {
        return _matmul_impl(out, tensor1, tensor2.squeeze(0));
      }
    }

    auto output_shape = infer_size_dimvector(batch_tensor1, batch_tensor2);
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expand_size = [&output_shape, n, m1]{ DimVector ret(output_shape);
                                                             ret.append({n, m1});
                                                             return ret; }();
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                         .reshape({expand_batch_product, n, m1});
    // We need to treat the dim_tensor2 == 1 case separately as broadcasting would not convert
    // a vector of shape (n,) into a batch of matrices of shape (*, n, 1)
    auto vector_rhs = dim_tensor2 == 1;
    const auto tensor2_expand_size = [&output_shape, m2, p, vector_rhs]{
      DimVector ret(output_shape);
      if (vector_rhs) {
        ret.push_back(m2);
      } else {
        ret.append({m2, p});
      }
      return ret;
    }();
    auto tensor2_expanded = tensor2.expand(tensor2_expand_size);
    if (vector_rhs) {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
    } else {
      tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2, p});
    }

    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    if (!has_out) {
      if (vector_rhs) {
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded).squeeze(-1), output_shape);
      } else {
        return at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);
      }
    } else {
      at::native::resize_output(out, output_shape);
      auto reshaped_out = out.reshape({expand_batch_product, n, p});
      at::bmm_out(reshaped_out, tensor1_expanded, tensor2_expanded);
      if (vector_rhs) {
        reshaped_out = reshaped_out.squeeze(-1);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out.view_as(out));
      }
      return out;
    }
  }
}

Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::Tensor result, unused;
  result = at::native::_matmul_impl(unused, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::native::_matmul_impl(result, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

// torch.linalg.matmul, alias for torch.matmul
Tensor linalg_matmul(const Tensor & tensor1, const Tensor & tensor2) {
  return at::matmul(tensor1, tensor2);
}

Tensor& linalg_matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  return at::matmul_out(result, tensor1, tensor2);
}

// torch.linalg.diagonal, alias for torch.diagonal with dim1=-2, dim2=-1 as defaults
Tensor linalg_diagonal(const Tensor& A, int64_t offset, int64_t dim1, int64_t dim2) {
  return A.diagonal(offset, dim1, dim2);
}

// helper methods for matrix_exp
namespace {

template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

// we consider 6 Taylor expansions of degree
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

Tensor operator_1_norm(const Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

// Allocates a buffers of uninitialized or zero values
// of shape [n_copies, a.size()]
Tensor _allocate_buffer(const Tensor& a, int n_copies, bool is_zero = false) {
  auto res = at::empty(
    {n_copies, a.size(0), a.size(1), a.size(2)},
    a.options().memory_format(at::MemoryFormat::Contiguous)
  );

  if (is_zero) {
    res.zero_();
  }

  return res;
}

// Makes `buffer` to store `num_matrices` number of matrices needed for
// compute the matrix exponentials of different orders, i.e.
// first `num_matrices` matrices from the list l := {I, A, A^2, A^3, A^6}
// in a contiguous block of memory such that
// buffer[0, ...] = l[0], // I
// buffer[1, ...] = l[1], // A
// ...
// buffer[num_matrices - 1, ...] = l[num_matries - 1]
void _fill_matrix_powers(Tensor& buffer, const Tensor& a, int num_matrices) {
  auto a_sizes_minus_last = a.sizes().vec();
  a_sizes_minus_last.pop_back();
  // fill I
  buffer.select(0, 0).copy_(
    at::diag_embed(
      at::ones({1}, buffer.options())
        .expand(a_sizes_minus_last)
    )
  );

  // fill a
  buffer.select(0, 1).copy_(a);

  // fill a^2
  if (2 <= num_matrices - 1) {
    // out for a^2
    auto view_out = buffer.select(0, 2);
    _matmul_impl(
      view_out,
      buffer.select(0, 1),
      buffer.select(0, 1)
    );
  }

  // fill a^3
  if (3 <= num_matrices - 1) {
    // out for a^3
    auto view_out = buffer.select(0, 3);
    _matmul_impl(
      view_out,
      buffer.select(0, 1),
      buffer.select(0, 2)
    );
  }

  // fill a^6
  if (4 <= num_matrices - 1) {
    // out for a^6
    auto view_out = buffer.select(0, 4);
    _matmul_impl(
      view_out,
      buffer.select(0, 3),
      buffer.select(0, 3)
    );
  }
}

inline Tensor _move_memory_if_cuda_input(
  const Tensor& mem,
  const Tensor& in
) {
  return (in.device().type() == at::kCUDA)
    ? mem.to(at::device_of(in).value())
    : mem;
}

// convert a 1D blob to a 2D Tensor of size [1, blob.size()]
// such that blob.device() == in.device())
// designed to be used with _compute_linear_combination
template <typename scalar_t>
inline Tensor _blob_to_Tensor(
  std::initializer_list<scalar_t> blob,
  const Tensor& in
) {
  // we convert to void* expecitly because begin() returns
  // a pointer to a constant.
  // Blob is assumed to be a 1D array, that is why
  // we also insert a fake dimension so that the result could directly
  // be used in _compute_linear_combination
  auto tensor = at::from_blob((void*)blob.begin(), blob.size(),
    c10::toRealValueType(in.scalar_type())).unsqueeze(0);
  return _move_memory_if_cuda_input(tensor, in);
}

template <typename scalar_t>
inline Tensor _linear_combination(
    const Tensor& t,
    std::initializer_list<scalar_t> blob) {
  // _blob_to_Tensor converts blob to a 2D tensor for _compute_linear_combination.
  // If this tensor is of shape (1, *), the result of _compute_linear_combination
  // is going to be of shape (1, *t.shape) so we squeeze(0) so that
  // for any t with t.dim() >= 1: t.dim() == _compute_linear_combination(t, ...).dim().
  return at::native::_compute_linear_combination(
      t, _blob_to_Tensor<scalar_t>(blob, t))
    .squeeze(0);
}

// I + A
Tensor compute_T1(const Tensor& A) {
  // 2 for {I, A}
  auto As = _allocate_buffer(A, 2);
  _fill_matrix_powers(As, A, 2);
  return As.sum(0);
}

// I + A + A^2 / 2
Tensor compute_T2(const Tensor& A) {
  auto As = _allocate_buffer(A, 3);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);
  As.select(0, 2).div_(2.0);
  return As.sum(0);
}

// I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
template <typename scalar_t>
Tensor compute_T4(const Tensor& A) {
  auto As = _allocate_buffer(A, 4);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);

  // output for A^2 * (I / 2 + A / 6 + A^2 / 24)
  auto view_out = As.select(0, 3);
  _matmul_impl(
    view_out,
    // contains A^2
    As.select(0, 2),
    // computes (I / 2 + A / 6 + A^2 / 24)
    _linear_combination<scalar_t>(
      As.narrow(0, 0, 3),
      {1 / 2.0, 1 / 6.0, 1 / 24.0}
    )
  );

  // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
  return _linear_combination<scalar_t>(
    As, {1.0, 1.0, 0.0, 1.0}
  );
}

template <typename scalar_t>
Tensor compute_T8(const Tensor& A) {
  constexpr scalar_t sqrt_177 = 0.1330413469565007072504e+2;
  constexpr scalar_t x3 = 2. / 3.;
  constexpr scalar_t x1 = x3 * ((1. + sqrt_177) / 88.);
  constexpr scalar_t x2 = x3 * ((1. + sqrt_177) / 352.);
  constexpr scalar_t x4 = (-271. + 29. * sqrt_177) / (315. * x3);
  constexpr scalar_t x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
  constexpr scalar_t x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
  constexpr scalar_t x7 = (89. - sqrt_177) / (5040. * x3);
  constexpr scalar_t y2 = (857. - 58. * sqrt_177) / 630.;

  auto As = _allocate_buffer(A, 5);
  // 3 for {I, A, A^2}
  _fill_matrix_powers(As, A, 3);

  // output for A4
  auto view_out = As.select(0, 3);
  // A4 =  A2 * (x1 * A + x2 * A2)
  _matmul_impl(
    view_out,
    // As.select(0, 2) = A^2
    As.select(0, 2),
    _linear_combination<scalar_t>(
      // extract {A, A^2} from As
      As.narrow(0, 1, 2),
      {x1, x2}
    )
  );

  // output for A8
  view_out = As.select(0, 4);
  // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
  _matmul_impl(
    view_out,
    // x3 * A2 + A4
    _linear_combination<scalar_t>(
      As.narrow(0, 2, 2),
      {x3, 1.0}
    ),
    _linear_combination<scalar_t>(
      As.narrow(0, 0, 4),
      {x4, x5, x6, x7}
    )
  );

  // return I + A + y2 * A2 + A8;
  return _linear_combination<scalar_t>(
    As, {1.0, 1.0, y2, 0.0, 1.0}
  );
}

template <typename scalar_t>
Tensor compute_T12(const Tensor& A) {
  constexpr int num_prods = 4;
  array2d<scalar_t, num_prods, num_prods> b = {{
    {
      9.0198e-16,
      0.46932117595418237389,
      -0.20099424927047284052,
      -0.04623946134063071740
    },
    {
      5.31597895759871264183,
      1.19926790417132231573,
      0.01179296240992997031,
      0.01108844528519167989
    },
    {
      0.18188869982170434744,
      0.05502798439925399070,
      0.09351590770535414968,
      0.00610700528898058230
    },
    {
      -2.0861320e-13,
      -0.13181061013830184015,
      -0.02027855540589259079,
      -0.00675951846863086359
    }
  }};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
    reinterpret_cast<void*>(&b),
    {num_prods, num_prods},
    {num_prods, 1},
    c10::toRealValueType(A.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = at::native::_compute_linear_combination(As, bs);

  // output for A6
  auto view_out = As.select(0, 0);
  // compute A6
  Bs.select(0, 2).add_(_matmul_impl(
    view_out,
    Bs.select(0, 3),
    Bs.select(0, 3)
  ));

  return Bs.select(0, 0).add_(_matmul_impl(
    view_out,
    Bs.select(0, 1).add_(Bs.select(0, 2)),
    Bs.select(0, 2)
  ));
}

template <typename scalar_t>
Tensor compute_T18(const Tensor& A) {
  constexpr int num_prods = 5;
  array2d<scalar_t, num_prods, num_prods> b = {{
    {
      0.,
      -1.00365581030144618291e-01,
      -8.02924648241156932449e-03,
      -8.92138498045729985177e-04,
      0.
    },
    {
      0.,
      3.97849749499645077844e-01,
      1.36783778460411720168e+00,
      4.98289622525382669416e-01,
      -6.37898194594723280150e-04
    },
    {
      -1.09676396052962061844e+01,
      1.68015813878906206114e+00,
      5.71779846478865511061e-02,
      -6.98210122488052056106e-03,
      3.34975017086070470649e-05
    },
    {
      -9.04316832390810593223e-02,
      -6.76404519071381882256e-02,
      6.75961301770459654925e-02,
      2.95552570429315521194e-02,
      -1.39180257516060693404e-05
    },
    {
      0.,
      0.,
      -9.23364619367118555360e-02,
      -1.69364939002081722752e-02,
      -1.40086798182036094347e-05
    }
  }};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
    reinterpret_cast<void*>(&b),
    {num_prods, num_prods},
    {num_prods, 1},
    c10::toRealValueType(A.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = at::native::_compute_linear_combination(As, bs);

  // tmp buffer for this matrix product
  auto view_out = As.select(0, 0);
  // compute A9
  Bs.select(0, 3).add_(_matmul_impl(
    view_out,
    Bs.select(0, 0),
    Bs.select(0, 4))
  );

  return Bs.select(0, 1).add_(_matmul_impl(
    view_out,
    Bs.select(0, 2).add_(Bs.select(0, 3)),
    Bs.select(0, 3)
  ));
}

template <typename scalar_t>
void compute_T18_scale_square(
  Tensor& mexp_out,
  const Tensor& a,
  const Tensor& norm,
  scalar_t theta
) {
  // Scale
  const auto s = at::max(
    at::zeros_like(norm),
    at::ceil(at::log2(norm / theta))
  ).unsqueeze(-1).unsqueeze(-1).to(at::kLong);
  const auto pow2s = at::pow(2, s);
  const auto a_scaled = a / pow2s;

  // Square
  auto mexp_scaled = at::native::compute_T18<scalar_t>(a_scaled);
  auto s_cpu = (s.device().type() == at::kCPU)
    ? s : s.to(at::kCPU);
  for (const auto i : c10::irange(mexp_scaled.size(0))) {
    auto s_val = s_cpu.select(0, i).template item<int64_t>();
    auto mexp = mexp_scaled.select(0, i);
    for (const auto p C10_UNUSED : c10::irange(s_val)) {
      mexp = at::matmul(mexp, mexp);
    }
    mexp_out.select(0, i).copy_(mexp);
  }
}

template <typename scalar_t>
Tensor mexp_impl(
  const Tensor& a,
  std::array<scalar_t, total_n_degs> thetas,
  bool compute_highest_degree_approx = false
) {
  auto res = at::empty_like(a);
  const auto norm = operator_1_norm(a);
  // `norm_cpu` is used to decide which Tensors require which approximation
  // based on their norm. This decision takes place on CPU.
  // It requires moving data back and forth between devices when `a` is on CUDA,
  // but at the cost of only one sigle CPU-CUDA synchronization (instead of 6),
  // and better performance overall (benchmarked).
  const auto norm_cpu = (a.device().type() == at::kCUDA)
    ? norm.to(at::kCPU) : norm;

  if (!compute_highest_degree_approx) {
    constexpr std::array<
      Tensor(*)(const Tensor&),
      total_n_degs - 1>
    compute_Ts = {
      compute_T1, compute_T2, compute_T4<scalar_t>,
      compute_T8<scalar_t>, compute_T12<scalar_t>
    };

    for (int i = 0; i < total_n_degs - 1; ++i) {
      auto norm_lower_bound = (i == 0) ? static_cast<scalar_t>(-1) : thetas[i - 1];
      auto norm_upper_bound = thetas[i];
      // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
      auto idx_curr_norm_interval = (
        (norm_lower_bound < norm_cpu) * (norm_cpu <= norm_upper_bound)
      ).nonzero().squeeze(-1);

      if (idx_curr_norm_interval.numel()) {
        auto idx_to_device = _move_memory_if_cuda_input(
          idx_curr_norm_interval, a
        );
        auto sub_a = at::index_select(a, 0, idx_to_device);
        res.index_put_({idx_to_device}, compute_Ts[i](sub_a));
      }
    }

    // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
    auto idx_large_norm = (norm_cpu >= thetas[total_n_degs - 2])
      .nonzero().squeeze(-1);

    if (idx_large_norm.numel()) {
      auto idx_to_device = _move_memory_if_cuda_input(
        idx_large_norm, a
      );
      auto a_large_norm = at::index_select(a, 0, idx_to_device);
      auto large_norm_subset = at::index_select(norm, 0, idx_to_device);
      auto mexp_out = at::empty_like(a_large_norm);

      compute_T18_scale_square(
        mexp_out,
        a_large_norm,
        large_norm_subset,
        thetas[total_n_degs - 1]
      );
      res.index_put_({idx_large_norm}, mexp_out);
    }

    return res;
  }

  compute_T18_scale_square(
    res, a, norm,
    thetas[total_n_degs - 1]
  );

  return res;
}

// matrix exponential
Tensor mexp(const Tensor& a, bool compute_highest_degree_approx = false) {
  // squash batch dimensions to one dimension for simplicity
  const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});

  if (a.scalar_type() == at::ScalarType::Float
      || a.scalar_type() == at::ScalarType::ComplexFloat) {
    constexpr std::array<float, total_n_degs> thetas_float = {
      1.192092800768788e-07, // deg 1
      5.978858893805233e-04, // deg 2
      5.116619363445086e-02, // deg 4
      5.800524627688768e-01, // deg 8
      1.461661507209034e+00, // deg 12
      3.010066362817634e+00  // deg 18
    };

    return mexp_impl<float>(a_3d, thetas_float, compute_highest_degree_approx)
      .view(a.sizes());
  }
  else { // if Double or ComplexDouble
    constexpr std::array<double, total_n_degs> thetas_double = {
      2.220446049250313e-16, // deg 1
      2.580956802971767e-08, // deg 2
      3.397168839976962e-04, // deg 4
      4.991228871115323e-02, // deg 8
      2.996158913811580e-01, // deg 12
      1.090863719290036e+00  // deg 18
    };

    return mexp_impl<double>(a_3d, thetas_double, compute_highest_degree_approx)
      .view(a.sizes());
  }
}

// TODO This should be deprecated in favor of linalg_matrix_exp_differential
//      in FunctionsManual.cpp
template <typename func_t>
Tensor backward_analytic_function_of_a_matrix(
    const Tensor& self, const Tensor& grad,
    const func_t& function_of_a_matrix
  ) {
  auto self_transposed = self.mH();
  auto self_transposed_sizes = self_transposed.sizes().vec();
  self_transposed_sizes[self.dim() - 2] <<= 1;
  self_transposed_sizes[self.dim() - 1] <<= 1;

  auto n = self_transposed.size(-1);
  auto meta_grad = at::zeros(self_transposed_sizes, grad.options());
  meta_grad.narrow(-2, 0, n).narrow(-1, 0, n).copy_(self_transposed);
  meta_grad.narrow(-2, n, n).narrow(-1, n, n).copy_(self_transposed);
  meta_grad.narrow(-2, 0, n).narrow(-1, n, n).copy_(grad);

  auto grad_input = function_of_a_matrix(meta_grad)
    .narrow(-2, 0, n).narrow(-1, n, n);
  return grad_input;
}
} // end anon namespace

// Computes the matrix exponential for a given batch of squared matrices.
// The implementaion is based on:
//
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
// Mathematics 2019, 7, 1174.
//
Tensor linalg_matrix_exp(const Tensor& a) {
  squareCheckInputs(a, "linalg.matrix_exp");
  checkFloatingOrComplex(a, "matrix_exp");

  NoTF32Guard disable_tf32;

  // Trivial cases
  const auto n = a.size(-1);
  if (n == 0) {
    return a.clone();
  } else if (n == 1) {
    return a.exp();
  } else {
    return at::native::mexp(a);
  }
}

// Alias
Tensor matrix_exp(const Tensor& a) {
  return at::linalg_matrix_exp(a);
}

// TODO This should be deprecated in favor of linalg_matrix_exp_differential
//      in FunctionsManual.cpp
Tensor matrix_exp_backward(const Tensor& self, const Tensor& grad) {
  NoTF32Guard disable_tf32;
  return backward_analytic_function_of_a_matrix(
    self, grad,
    [](const Tensor& a) {
      return a.matrix_exp();
    }
  );
}

TORCH_IMPL_FUNC(linalg_vector_norm_out)(const Tensor& self, const Scalar& scalar_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype, const Tensor& result) {
  // Casting a large integer to a double will just introduce an error for
  // values larger than 10^53 (same for negative numbers), so that's fine.
  auto ord = scalar_ord.toDouble();
  auto dim = opt_dim.value_or(IntArrayRef{});
  // No need to handle opt_dtype explicitly as it is already encoded in the dtype of result

  // https://github.com/pytorch/pytorch/issues/52648
  // Reductions always use `std::abs` to compute the absolute value. In the backward of this
  // function, we need to locate the index that was selected as the largest value. To do so
  // we do self.abs() == result to locate the index of the largest element.
  // Now, self.abs() may dispatch to a vectorized implementation which gives sliiightly different
  // results to the std::abs(std::complex<T>) implementation.
  // As such, to be able to compute the correct index in the backward, we need to use self.abs()
  // both in the forward and in the backward
  Tensor self_;
  if (self.is_cpu() && self.is_complex() && std::abs(ord) == INFINITY) {
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype).abs();
    } else {
      self_ = self.abs();
    }
  } else {
    self_ = self;
  }

  auto iter = make_reduction("vector_norm", const_cast<Tensor&>(result), self_, dim, keepdim, result.scalar_type());
  norm_stub(iter.device_type(), iter, ord);
}

void _linalg_matrix_norm_checks(const Tensor& A, std::vector<int64_t>& dim, optional<ScalarType> opt_dtype, bool low_precision) {
  // A
  at::native::checkIsMatrix(A, "linalg.matrix_norm");
  at::native::checkFloatingOrComplex(A, "linalg.matrix_norm", /*low_precision*/low_precision);

  // dim
  TORCH_CHECK(dim.size() == 2, "linalg.matrix_norm: dim must be a 2-tuple. Got ", dim);
  // wrap first to identify weird scenarios like A.ndim = 2, dim = (1, -1)
  // dim is modified in place while wrapping it
  maybe_wrap_dims(dim, A.dim());
  TORCH_CHECK(dim[0] != dim[1], "linalg.matrix_norm: dims must be different. Got (", dim[0], ", ", dim[1], ")");

  // dtype
  at::detail::check_linalg_norm_dtype(opt_dtype, A.scalar_type(), "linalg.matrix_norm");
}

Tensor linalg_matrix_norm(
    const Tensor& A,
    const Scalar& scalar_ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  // Check ord first as it will be used in the dtype check of A
  auto ord = scalar_ord.toDouble();
  auto abs_ord = std::abs(ord);
  TORCH_CHECK(abs_ord == 2. || abs_ord == 1. || abs_ord == INFINITY, "linalg.matrix_norm: Order ", ord, " not supported.");

  auto dim_ = dim.vec();
  // Check A, dim, and dtype
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, /*low_precision*/abs_ord != 2.);

  auto max_min = [ord, keepdim](const Tensor& A, int64_t dim) { return ord > 0 ? A.amax(dim, keepdim) : A.amin(dim, keepdim); };
  if (abs_ord == 2.) {
    // Move dims to the end
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A.dim());

    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;
    auto result = max_min(at::linalg_svdvals(A_.permute(permutation)), -1);
    if (keepdim) {
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;
  } else {  // 1, -1, inf, -inf
    // The infty norm is like the 1 norm on the transposed matrix
    if (abs_ord == INFINITY) {
      std::swap(dim_[0], dim_[1]);
    }

    // If the first reduction removes one dim from the front (dim_[0] < dim_[1]), after this
    // reduction dim_[1] will be off by one
    if (!keepdim && (dim_[0] < dim_[1])) {
      dim_[1]--;
    }
    return max_min(at::linalg_vector_norm(A, 1., {dim_[0]}, keepdim, opt_dtype), dim_[1]);
  }
}

Tensor& linalg_matrix_norm_out(
    const Tensor& A,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  checkSameDevice("linalg.matrix_norm", A, result);
  auto out = at::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.matrix_norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  at::native::resize_output(result, out.sizes());
  result.copy_(out);
  return result;
}

// fro / nuc
Tensor linalg_matrix_norm(
    const Tensor& A,
    c10::string_view ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  // Check ord first as it will be used in the dtype check of A
  TORCH_CHECK(ord == "fro" || ord == "nuc", "linalg.matrix_norm: Order ", ord, " not supported.");

  auto dim_ = dim.vec();
  // Check A, dim, and dtype
  _linalg_matrix_norm_checks(A, dim_, opt_dtype, /*low_precision*/ord != "nuc");

  if (ord == "fro") {
    return at::linalg_vector_norm(A, 2, dim_, keepdim, opt_dtype);
  } else {  // nuc
    auto A_ = opt_dtype.has_value() ? A.to(*opt_dtype) : A;

    // Move dims to the end
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], A_.dim());
    auto result = at::linalg_svdvals(A_.permute(permutation)).sum(-1, keepdim);
    if (keepdim) {
      auto permutation_reverse = create_reverse_permutation(std::move(permutation));
      result = result.unsqueeze(-1).permute(permutation_reverse);
    }
    return result;
  }
}

Tensor& linalg_matrix_norm_out(
    const Tensor& A,
    c10::string_view ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  checkSameDevice("linalg.matrix_norm", A, result);
  auto out = at::linalg_matrix_norm(A, ord, dim, keepdim, opt_dtype);
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.matrix_norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  at::native::resize_output(result, out.sizes());
  result.copy_(out);
  return result;
}

// Numerical or None norms
Tensor linalg_norm(const Tensor& X, const optional<Scalar>& opt_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim ->size() == 2, "linalg.norm: If ",
              "dim is specified, it must be of length 1 or 2. Got ", *opt_dim);
  } else {
    if (opt_ord.has_value()) {
      TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                  "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
    }
  }

  // If ord=None, we'll always use the 2-norm or frob norm (which are the same) so we go through
  // vector_norm
  if (opt_ord.has_value() &&
       ((opt_dim.has_value() && opt_dim->size() == 2) ||
        (!opt_dim.has_value() && X.dim() == 2))) {
    using Int = IntArrayRef::value_type;
    auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<Int>{0, 1};
    return at::linalg_matrix_norm(X, *opt_ord, dim, keepdim, opt_dtype);
  } else {
    auto scalar_ord = opt_ord.value_or(Scalar(2.));
    return at::linalg_vector_norm(X, scalar_ord, opt_dim, keepdim, opt_dtype);
  }
}

Tensor& linalg_norm_out(const Tensor& X, const optional<Scalar>& opt_ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  checkSameDevice("linalg.norm", X, result);
  auto out = at::linalg_norm(X, opt_ord, opt_dim, keepdim, opt_dtype);
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  at::native::resize_output(result, out.sizes());
  result.copy_(out);
  return result;
}

// Frobenius and nuclear norms
Tensor linalg_norm(const Tensor& X, c10::string_view ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  if (opt_dim.has_value()) {
    TORCH_CHECK(opt_dim->size() == 1 || opt_dim ->size() == 2, "linalg.norm: If ",
              "dim is specified, it mut be of length 1 or 2. Got ", *opt_dim);
  } else {
    TORCH_CHECK(X.dim() == 1 || X.dim() == 2, "linalg.norm: If ",
                "dim is not specified but ord is, the input must be 1D or 2D. Got ", X.dim(), "D.");
  }
  using Int = IntArrayRef::value_type;
  auto dim = opt_dim.has_value() ? opt_dim.value().vec() : std::vector<Int>{0, 1};
  return at::linalg_matrix_norm(X, ord, dim, keepdim, opt_dtype);
}

Tensor& linalg_norm_out(const Tensor& X, c10::string_view ord, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  checkSameDevice("linalg.norm", X, result);
  auto out = at::linalg_norm(X, ord, opt_dim, keepdim, opt_dtype);
  TORCH_CHECK(out.scalar_type() == result.scalar_type(),
              "linalg.norm expected out tensor dtype ", out.scalar_type(),
              " but got: ", result.scalar_type());
  at::native::resize_output(result, out.sizes());
  result.copy_(out);
  return result;
}

////////////////////////////////////////////////////////////////////////////////
//                              Frobenius Norm                                //
////////////////////////////////////////////////////////////////////////////////

Tensor frobenius_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto device = self.device();
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    TORCH_WARN_ONCE(
      "at::frobenius_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.vector_norm(A, 2., dim, keepdim)` instead"
    );
  }
  // This frobenius norm is just wrong, but well
  TORCH_CHECK(dim.size() <= 2,
              "Expected at most 2 dimensions, but got ", dim.size(), " dimensions instead.");
  // Dispatch to at::norm as it is implemented for Sparse and MPS backends
  // TODO Make the backends implement vector_norm and matrix_norm
  return at::norm(self, 2., dim, keepdim);
}

Tensor &frobenius_norm_out(const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    Tensor& result) {
  auto device = self.device();
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    TORCH_WARN_ONCE(
      "at::frobenius_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.vector_norm(A, 2., dim, keepdim)` instead"
    );
  }
  TORCH_CHECK(dim.size() <= 2,
              "Expected at most 2 dimensions, but got ", dim.size(), " dimensions instead.");
  return at::norm_out(result, self, 2., dim, keepdim);
}

////////////////////////////////////////////////////////////////////////////////
//                                Nuclear Norm                                //
////////////////////////////////////////////////////////////////////////////////

Tensor nuclear_norm(const Tensor& self, bool keepdim) {
  return at::native::nuclear_norm(self, IntArrayRef({-2, -1}), keepdim);
}

Tensor &nuclear_norm_out(const Tensor& self, bool keepdim, Tensor& result) {
  auto device = self.device();
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  return at::linalg_matrix_norm_out(result, self, "nuc", IntArrayRef({-2, -1}), keepdim);
}

Tensor nuclear_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  auto device = self.device();
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  return at::linalg_matrix_norm(self, "nuc", dim, keepdim);
}

Tensor& nuclear_norm_out(const Tensor& self, IntArrayRef dim, bool keepdim, Tensor& result) {
  auto device = self.device();
  if (self.layout() == Layout::Strided && (device == kCPU || device == kCUDA || device == kMeta)) {
    TORCH_WARN_ONCE(
      "at::nuclear_norm is deprecated and it is just left for JIT compatibility. ",
      "It will be removed in a future PyTorch release. Please use ",
      "`linalg.matrix_norm(A, 'nuc', dim, keepdim)` instead"
    );
  }
  return at::linalg_matrix_norm_out(result, self, "nuc", dim, keepdim);
}

////////////////////////////////////////////////////////////////////////////////
//                              linalg.cond                                   //
////////////////////////////////////////////////////////////////////////////////


// This function helps to dispatch norm computations depending on 'ord' of variant type
Tensor _linalg_cond_helper(const Tensor& self, c10::variant<Scalar, c10::string_view> ord_variant) {
  Tensor inverse, info;
  std::tie(inverse, info) = at::linalg_inv_ex(self);
  info.unsqueeze_(-1).unsqueeze_(-1);
  inverse.masked_fill_(info > 0, INFINITY);

  return c10::visit([&](auto&& ord) {
    Tensor norm_self = at::linalg_matrix_norm(self, ord);
    Tensor norm_inverse = at::linalg_matrix_norm(inverse, ord);
    Tensor result = norm_self * norm_inverse;
    // fix multiplication of zero and infinity for NumPy compatibility
    result.nan_to_num_(INFINITY, INFINITY, -INFINITY);
    return result;
  }, ord_variant);
}

// Return zero for each matrix in the batch
Tensor _linalg_cond_empty_matrix(const Tensor& self, c10::ScalarType dtype) {
  auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
  TensorOptions options = self.options().dtype(toRealValueType(self.scalar_type()));
  return at::zeros(result_shape, options);
}

void _linalg_cond_check_ord(c10::variant<Scalar, c10::string_view> ord_variant) {
  if (ord_variant.index() == 0) {
    Scalar* ord = c10::get_if<Scalar>(&ord_variant);
    double abs_ord = std::abs(ord->toDouble());
    TORCH_CHECK(abs_ord == 2.0 || abs_ord == 1.0 || abs_ord == INFINITY,
      "linalg.cond got an invalid norm type: ", ord->toDouble());
  } else if (ord_variant.index() == 1) {
    c10::string_view* ord = c10::get_if<c10::string_view>(&ord_variant);
    TORCH_CHECK(*ord == "fro" || *ord == "nuc",
      "linalg.cond got an invalid norm type: ", *ord);
  } else {
    TORCH_CHECK(false,
      "linalg.cond: something went wrong while checking the norm type");
  }
}

// Numerical or None norms
Tensor linalg_cond(const Tensor& self, const optional<Scalar>& opt_ord) {
  TORCH_CHECK(self.dim() >= 2, "linalg.cond: The input tensor must have at least 2 dimensions.");

  // The default case is using 2-norm
  Scalar ord = opt_ord.has_value() ? opt_ord.value() : 2;

  c10::variant<Scalar, c10::string_view> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
  if (self.sym_numel() == 0) {
    auto real_dtype = toRealValueType(typeMetaToScalarType(self.dtype()));
    return _linalg_cond_empty_matrix(self, real_dtype);
  }

  // If ord == None or ord == ±2
  if (std::abs(ord.toDouble()) == 2.0) {
    auto singular_values = at::linalg_svdvals(self);
    // singular values are sorted in descending order
    auto s_max = at::narrow(singular_values, /*dim=*/-1, /*start=*/0, /*length=*/1);
    auto s_min = at::narrow(singular_values, /*dim=*/-1, /*start=*/-1, /*length=*/1);
    Tensor result;
    if (ord.toDouble() == -2.0) {
      result = s_min / s_max;
    } else {
      result = s_max / s_min;
    }
    // squeeze the result for NumPy compatibility
    return result.squeeze(-1);
  }

  // ord == ±1 ord == ±inf
  if (ord.isFloatingPoint()) { // ord == ±1
    squareCheckInputs(self, ("linalg.cond(ord=" + std::to_string(ord.to<double>()) + ")").c_str());
  } else { // ord == ±inf
    squareCheckInputs(self, ("linalg.cond(ord=" + std::to_string(ord.to<int64_t>()) + ")").c_str());
  }
  return _linalg_cond_helper(self, std::move(ord_variant));
}

Tensor& linalg_cond_out(const Tensor& self, const optional<Scalar>& opt_ord, Tensor& result) {
  checkSameDevice("linalg.cond", result, self);
  ScalarType real_dtype = toRealValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg.cond", result.scalar_type(), real_dtype);

  Tensor result_tmp = at::linalg_cond(self, opt_ord);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

// Frobenius or nuclear norms
Tensor linalg_cond(const Tensor& self, c10::string_view ord) {
  squareCheckInputs(self, ("linalg.cond(ord=" + std::string(ord) + ")").c_str());
  c10::variant<Scalar, c10::string_view> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
  if (self.numel() == 0) {
    return _linalg_cond_empty_matrix(self, self.scalar_type());
  }

  if (ord == "nuc") {
    // calling matrix_norm with "nuc" on inputs with infinities raises an error
    // therefore we use the mathematical definition of nuclear norm directly
    // instead of going through the matrix_norm
    auto singular_values = at::linalg_svdvals(self);
    return singular_values.sum(-1) * (singular_values.reciprocal().sum(-1));
  }

  return _linalg_cond_helper(self, std::move(ord_variant));
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_cond_out(const Tensor& self, c10::string_view ord, Tensor& result) {
  checkSameDevice("linalg.cond", result, self);
  ScalarType real_dtype = toRealValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg.cond", result.scalar_type(), real_dtype);

  Tensor result_tmp = at::linalg_cond(self, ord);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor linalg_tensorinv(const Tensor& self, int64_t ind) {
  /*
  The idea is to reduce the problem to 2D square matrix inversion.
  Step 1. Calculate the shape of the result and the shape of the intermediate 2D matrix.
  Step 2. Reshape `self` to 2D matrix.
  Step 3. Invert the 2D matrix self.to_2D()
          There is no quick way to find out whether the matrix is invertible,
          so at this stage an error from at::inverse can be thrown.
          Note that for CUDA this causes cross-device memory synchronization that can be slow.
  Step 4. reshape the result.
  */
  TORCH_CHECK(ind > 0, "Expected a strictly positive integer for 'ind', but got ", ind);

  // self[ind:]
  std::vector<c10::SymInt> shape_ind_end = self.sym_sizes().slice(ind).vec();
  // self[:ind]
  std::vector<c10::SymInt> shape_start_ind = self.sym_sizes().slice(0, ind).vec();

  c10::SymInt prod_ind_end = c10::multiply_integers(shape_ind_end.cbegin(), shape_ind_end.cend());
  c10::SymInt prod_start_ind = c10::multiply_integers(shape_start_ind.cbegin(), shape_start_ind.cend());

  // Check whether the self tensor can be reshaped to the 2D square matrix
  TORCH_CHECK(prod_ind_end == prod_start_ind,
    "Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got ",
    prod_ind_end, " != ", prod_start_ind);

  // Concatenate shape_ind_end and shape_start_ind to form the shape of the result
  // self[ind:] + self[:ind]
  shape_ind_end.insert(shape_ind_end.cend(), shape_start_ind.cbegin(), shape_start_ind.cend());

  // If the reshaped self is not invertible catch this error
  auto [result, info] = at::linalg_inv_ex(self.reshape_symint({prod_ind_end, prod_ind_end}), /*check_errors=*/false);
  at::_linalg_check_errors(info, "inv", /*is_matrix*/true);

  return result.reshape_symint(shape_ind_end);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_tensorinv_out(const Tensor& self, int64_t ind, Tensor& result) {
  checkSameDevice("tensorinv", result, self);
  checkLinalgCompatibleDtype("tensorinv", result, self);

  Tensor result_tmp = at::linalg_tensorinv(self, ind);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor linalg_tensorsolve(const Tensor& self, const Tensor& other, OptionalIntArrayRef dims) {
  /*
  The idea is to reduce the problem to 2D matrix solve.
  Step 1. (optional) `self` is permuted with `dims` such that dimensions from `dims` are moved to the right.
  For example, if we have 4D input with the shape (1, 2, 3, 4) and dims=(0, 2),
  then the result of permutation would have the shape (2, 4, 1, 3).
  Step 2. reshape `self` to 2D matrix.
  Step 3. solve the matrix equation self.to_2D() @ result = other.to_1D()
  Step 4. reshape the result.
  */
  int64_t ndim = self.dim();
  Tensor self_ = self;

  // move dimensions of `self_` from `dims` to the end
  if (dims.has_value()) {
    DimVector dest_axes(dims.value().size());
    std::iota(dest_axes.begin(), dest_axes.end(), ndim - dest_axes.size());
    self_ = at::movedim(self_, dims.value(), dest_axes);
  }

  // result_shape is self_.sizes[-(an-other.dim):]
  std::vector<c10::SymInt> result_shape = self_.sym_sizes().slice(other.dim(), ndim - other.dim()).vec();

  c10::SymInt result_product = c10::multiply_integers(result_shape.begin(), result_shape.end());
  c10::SymInt other_product = c10::multiply_integers(other.sym_sizes().begin(), other.sym_sizes().end());

  // Check whether the self tensor can be reshaped to the 2D square matrix
  TORCH_CHECK(result_product == other_product,
    "Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got ",
    result_product, " != ", other_product);

  self_ = self_.reshape_symint({result_product, result_product});

  // normally `other` would be flattened by at::linalg_solve expects 2D input
  Tensor result = at::linalg_solve(self_, other.flatten());
  return result.reshape_symint(result_shape);
}

Tensor& linalg_tensorsolve_out(const Tensor& self, const Tensor& other, OptionalIntArrayRef dims, Tensor& result) {
  checkSameDevice("tensorsolve", result, self);
  checkLinalgCompatibleDtype("tensorsolve", result, self);

  Tensor result_tmp = at::linalg_tensorsolve(self, other, dims);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

namespace {
struct KronImpl final {
  public:
    explicit KronImpl(const Tensor& self, const Tensor& other) {
      maxdim = std::max(self.dim(), other.dim());
      int64_t pad_self = maxdim - self.dim();
      int64_t pad_other = maxdim - other.dim();
      a_reshape = c10::SmallVector<int64_t, 10>(2 * maxdim);
      b_reshape = c10::SmallVector<int64_t, 10>(2 * maxdim);
      result_reshape = c10::SmallVector<int64_t, 10>(maxdim);
      for (const auto i : c10::irange(maxdim)) {
        a_reshape[2 * i] = (i >= pad_self ? self.sizes()[i - pad_self] : 1);
        a_reshape[2 * i + 1] = 1;
        b_reshape[2 * i] = 1;
        b_reshape[2 * i + 1] = (i >= pad_other ? other.sizes()[i - pad_other] : 1);
        result_reshape[i] = a_reshape[2 * i] * b_reshape[2 * i + 1];
      }
      self_view = at::_unsafe_view(self, a_reshape);
      other_view = at::_unsafe_view(other, b_reshape);
    }

    Tensor& kron_out(Tensor& result) const {
      TORCH_INTERNAL_ASSERT(result.defined(), "Cannot call kron_out with an undefined result tensor as the out argument. Please allocate a Tensor before calling kron_out with it.");

      c10::SmallVector<int64_t, 10> mul_shape(2 * maxdim);
      for (const auto i : c10::irange(maxdim)) {
        mul_shape[2 * i] = a_reshape[2 * i];
        mul_shape[2 * i + 1] = b_reshape[2 * i + 1];
      }
      at::native::resize_output(result, result_reshape);
      auto result_mul = at::_unsafe_view(result, mul_shape);
      at::mul_out(result_mul, self_view, other_view);

      return result;
    }

    Tensor kron() const {
      return at::_unsafe_view(at::mul(self_view, other_view), result_reshape);
    }
  private:
    int64_t maxdim;
    Tensor self_view;
    Tensor other_view;
    c10::SmallVector<int64_t, 10> result_reshape;
    c10::SmallVector<int64_t, 10> a_reshape;
    c10::SmallVector<int64_t, 10> b_reshape;
};
}

/*
Calculates the Kronecker product between two Tensors.
*/
Tensor& kron_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return KronImpl(self, other).kron_out(result);
}

Tensor kron(const Tensor& self, const Tensor& other) {
  return KronImpl(self, other).kron();
}

} // namespace native
} // namespace at
