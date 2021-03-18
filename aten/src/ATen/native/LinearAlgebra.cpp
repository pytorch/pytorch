#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>
#include <c10/util/variant.h>

#include <functional>
#include <limits>
#include <numeric>
#include <vector>


namespace at {
namespace native {

DEFINE_DISPATCH(addr_stub);
DEFINE_DISPATCH(linalg_vector_norm_stub);

// Helper function for det methods.
// For pivoted LU factorization A = P * L * U. Since we always have det(L) = 1,
// det(P) = \pm 1, this method returns a 3-tuple:
//   (det(P), diag(U), info),
// where info helps us identify singular matrices.
static inline std::tuple<Tensor, Tensor> _lu_det_P_diag_U(const Tensor& self) {
  Tensor pivs, lu, infos;
  std::tie(lu, pivs, infos) = at::_lu_with_info(self, /*pivot=*/true, /*check_errors=*/false);
  TORCH_CHECK(infos.ge(0).all().item<uint8_t>(), "Invalid argument passed to lu");
  auto n = self.size(-1);
  auto num_exchanges = (at::arange(1, n + 1, pivs.options()) != pivs)
    .sum(-1, /*keepdim=*/false, /*dtype=*/at::kLong).fmod_(2);
  auto u_diagonal = lu.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  return std::tuple<Tensor, Tensor>(num_exchanges.mul_(-2).add_(1), u_diagonal);
}

// torch.linalg.det, alias for torch.det
Tensor linalg_det(const Tensor& self) {
  return self.det();
}

Tensor det(const Tensor& self) {
  squareCheckInputs(self);
  TORCH_CHECK((at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type())),
              "Expected a floating point tensor as input");

  Tensor det_P, diag_U;
  std::tie(det_P, diag_U) = _lu_det_P_diag_U(self);
  // complete_det is 0 when U is singular (U(i, i) = 0 for some i in [1, self.size(-1)]).
  // The product accumulation takes care of this case, and hence no special case handling is required.
  auto complete_det = diag_U.prod(-1).mul_(det_P);
  return complete_det;
}

Tensor logdet(const Tensor& self) {
  squareCheckInputs(self);
  TORCH_CHECK((at::isFloatingType(self.scalar_type()) || at::isComplexType(self.scalar_type())),
              "Expected a floating point tensor as input");

  Tensor det_P, diag_U;
  std::tie(det_P, diag_U) = _lu_det_P_diag_U(self);
  Tensor det_sign = diag_U.sign().prod(-1).mul_(det_P);

  // If det_sign > 0, diag_U.abs_().log_().sum(-1) gives logdet (this means U is not singular).
  // If det_sign <= 0, then we get proper nan (when det < 0, i.e., det_sign) or -inf (when det = 0, i.e., U is singular).
  // U is singular when U(i, i) = 0 for some i in [1, self.size(-1)].
  Tensor logdet_vals = diag_U.abs_().log_().sum(-1);
  if (self.dim() > 2) {
    auto indices = toListOfOptionalTensors((det_sign < 0).nonzero_numpy());
    logdet_vals.index_put_(std::move(indices), at::full({}, NAN, self.options()));
  } else if (det_sign.item<double>() < 0) {
    logdet_vals.fill_(NAN);
  }
  return logdet_vals;
}

std::tuple<Tensor, Tensor> linalg_slogdet(const Tensor& self) {
  squareCheckInputs(self);
  ScalarType t = self.scalar_type();
  TORCH_CHECK(t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble,
              "linalg_slogdet: expected a tensor of float, double, cfloat or cdouble types but got ", t);

  Tensor det_P, diag_U;
  std::tie(det_P, diag_U) = _lu_det_P_diag_U(self);
  auto det_sign = diag_U.sgn().prod(-1).mul_(det_P);
  // abslogdet_val is -inf if U is singular, in which case diag_U.abs_().log_().sum(-1) will return -inf.
  // U is singular when U(i, i) = 0 for some i in [1, self.size(-1)].
  // Since abslogdet_val cannot take nan, no special case handling is required.
  // in-place abs is not supported for complex tensors
  auto abslogdet_val = isComplexType(t) ? diag_U.abs().log_().sum(-1) : diag_U.abs_().log_().sum(-1);
  return std::make_tuple(det_sign, abslogdet_val);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
std::tuple<Tensor&, Tensor&> linalg_slogdet_out(const Tensor& input, Tensor& sign, Tensor& logabsdet) {
  checkSameDevice("linalg_slogdet", sign, input, "sign");
  checkSameDevice("linalg_slogdet", logabsdet, input, "logabsdet");
  checkLinalgCompatibleDtype("linalg_slogdet", sign, input, "sign");
  ScalarType real_dtype = toValueType(input.scalar_type());
  // logabsdet is always real-valued here
  checkLinalgCompatibleDtype("linalg_slogdet", logabsdet.scalar_type(), real_dtype, "logabsdet");

  Tensor sign_tmp, logabsdet_tmp;
  std::tie(sign_tmp, logabsdet_tmp) = at::linalg_slogdet(input);

  at::native::resize_output(sign, sign_tmp.sizes());
  sign.copy_(sign_tmp);
  at::native::resize_output(logabsdet, logabsdet_tmp.sizes());
  logabsdet.copy_(logabsdet_tmp);

  return std::tuple<Tensor&, Tensor&>(sign, logabsdet);
}

std::tuple<Tensor, Tensor> slogdet(const Tensor& self) {
  return at::linalg_slogdet(self);
}

Tensor linalg_pinv(const Tensor& input, const Tensor& rcond, bool hermitian) {
  NoTF32Guard disable_tf32;
  ScalarType t = input.scalar_type();
  TORCH_CHECK((t == ScalarType::Double || t == ScalarType::Float || t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble)
              && input.dim() >= 2,
              "linalg_pinv(", t, "{", input.sizes(), "}): expected a tensor with 2 or more dimensions "
              "of float, double, cfloat or cdouble types");
  TORCH_CHECK(rcond.device() == input.device(),
              "Expected rcond and input to be on the same device, but found rcond on ",
              rcond.device(), " and input on ", input.device(), " instead.");
  TORCH_CHECK(!at::isComplexType(rcond.scalar_type()),
              "linalg_pinv: rcond tensor of complex type is not supported.");

  if (input.numel() == 0) {
    // The implementation below uses operations that do not work for zero numel tensors
    // therefore we need this early return for 'input.numel() == 0' case
    Tensor U, S, V;
    // TODO: replace input.svd with linalg_svd when torch/xla can work with at::linalg_svd
    std::tie(U, S, V) = input.svd();
    return at::matmul(V * S.reciprocal().unsqueeze(-2), U.conj().transpose(-2, -1));
  }

  // If not Hermitian use singular value decomposition, else use eigenvalue decomposition
  if (!hermitian) {
    Tensor U, S, V;
    // TODO: replace input.svd with linalg_svd
    // using linalg_svd breaks pytorch/xla, see https://github.com/pytorch/xla/issues/2755
    std::tie(U, S, V) = input.svd();
    Tensor max_val = at::narrow(S, /*dim=*/-1, /*start=*/0, /*length=*/1);  // singular values are sorted in descending order
    Tensor S_pseudoinv = at::where(S > (rcond.unsqueeze(-1) * max_val), S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes V @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(V * S_pseudoinv.unsqueeze(-2), U.conj().transpose(-2, -1));
  } else {
    Tensor S, U;
    std::tie(S, U) = at::linalg_eigh(input);
    // For Hermitian matrices, singular values equal to abs(eigenvalues)
    Tensor S_abs = S.abs();
    // eigenvalues are sorted in ascending order starting with negative values, we need a maximum value of abs(eigenvalues)
    Tensor max_val = S_abs.amax(/*dim=*/-1, /*keepdim=*/true);
    Tensor S_pseudoinv = at::where(S_abs > (rcond.unsqueeze(-1) * max_val), S.reciprocal(), at::zeros({}, S.options())).to(input.dtype());
    // computes U @ diag(S_pseudoinv) @ U.conj().T
    return at::matmul(U * S_pseudoinv.unsqueeze(-2), U.conj().transpose(-2, -1));
  }
}

Tensor linalg_pinv(const Tensor& input, double rcond, bool hermitian) {
  Tensor rcond_tensor = at::full({}, rcond, input.options().dtype(ScalarType::Double));
  return at::linalg_pinv(input, rcond_tensor, hermitian);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_pinv_out(Tensor& result, const Tensor& input, const Tensor& rcond, bool hermitian) {
  checkSameDevice("linalg_pinv", result, input);
  checkLinalgCompatibleDtype("linalg_pinv", result, input);

  Tensor result_tmp = at::linalg_pinv(input, rcond, hermitian);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor& linalg_pinv_out(Tensor& result, const Tensor& input, double rcond, bool hermitian) {
  Tensor rcond_tensor = at::full({}, rcond, input.options().dtype(ScalarType::Double));
  return at::linalg_pinv_out(result, input, rcond_tensor, hermitian);
}

Tensor pinverse(const Tensor& self, double rcond) {
  return at::linalg_pinv(self, rcond, /*hermitian=*/false);
}

Tensor& linalg_matrix_rank_out(Tensor& result, const Tensor& self, optional<double> tol, bool hermitian) {
  checkSameDevice("linalg_matrix_rank", result, self);
  ScalarType output_type = ScalarType::Long;
  checkLinalgCompatibleDtype("linalg_matrix_rank", result.scalar_type(), output_type);

  // Matrices or batch of matrices are allowed
  TORCH_CHECK(self.dim() >= 2, "linalg_matrix_rank: Expected as input a matrix or a batch of matrices, but got a tensor of size: ", self.sizes());

  // matrix_rank assigns a scalar value for each matrix in the batch so
  // result's shape is equal to self.shape[0:self.ndim-2]
  // for single matrix result_shape = {}
  auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
  at::native::resize_output(result, result_shape);

  // NumPy doesn't take into account possible input with no elements and it errors on max not defined for this case
  // Let's output 0 for this case, since that kind of matrices have zero number of non-zero rows, hence rank is 0.
  if (self.numel() == 0) {
    result.fill_(0);
    return result;
  }

  // We compute matrix rank as the number of singular or absolute eigen values above 'tol' threshold
  Tensor S;
  if (!hermitian) {
    Tensor U, V;
    // TODO: replace self.svd with linalg_svd
    std::tie(U, S, V) = self.svd(/*some=*/true, /*compute_uv=*/false);
  } else {
    S = at::linalg_eigvalsh(self);
    S = S.abs();
  }

  if (tol.has_value()) {
    double tol_value = tol.value();
    at::sum_out(result, S > tol_value, /*dim=*/-1);
  } else {
    ScalarType real_dtype = toValueType(typeMetaToScalarType(self.dtype()));
    double tol_value = _get_epsilon(real_dtype) * std::max(self.size(-1), self.size(-2));
    Tensor max_S = S.amax(/*dim=*/-1);
    at::sum_out(result, S > max_S.mul_(tol_value).unsqueeze_(-1), /*dim=*/-1);
  }
  return result;
}

Tensor linalg_matrix_rank(const Tensor& self, optional<double> tol, bool hermitian) {
  Tensor result = at::empty({0}, self.options().dtype(ScalarType::Long));
  result = at::linalg_matrix_rank_out(result, self, tol, hermitian);
  return result;
}

Tensor matrix_rank(const Tensor& self, double tol, bool symmetric) {
  return at::linalg_matrix_rank(self, optional<double>(tol), symmetric);
}

Tensor matrix_rank(const Tensor& self, bool symmetric) {
  return at::linalg_matrix_rank(self, c10::nullopt, symmetric);
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

  Tensor self_;
  if (&result != &self) {
    std::tie(self_) = expand_size(self, {vec1.size(0), vec2.size(0)}, "addr");
  } else {
    self_ = self;
  }
  TORCH_CHECK(
    self_.dim() == 2,
    "2D tensor expected, got ", self_.dim(), "D tensor for input"
  );
  TORCH_CHECK(
    self_.size(0) == vec1.size(0) && self_.size(1) == vec2.size(0),
    "size mismatch, input: ", self_.sizes(),
    ", v1: ", vec1.sizes(),
    ", v2: ", vec2.sizes()
  );

  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(true)
    .add_output(result)
    .add_input(self_)
    .add_input(vec1.reshape({vec1.size(0), 1}))
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
  if (beta.toComplexDouble() == 0.0) {
    if (alpha.toComplexDouble() == 1.0) {
      return at::outer(vec1, vec2);
    }
    return alpha * at::outer(vec1, vec2);
  }

  if (beta.toComplexDouble() == 1.0) {
    if (alpha.toComplexDouble() == 1.0) {
      return self + at::outer(vec1, vec2);
    }
    return self + alpha * at::outer(vec1, vec2);
  }

  if (alpha.toComplexDouble() == 1.0) {
    return beta * self + at::outer(vec1, vec2);
  }
  return beta * self + alpha * at::outer(vec1, vec2);
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
Tensor& ger_out(Tensor &result, const Tensor& self, const Tensor& vec2) {
  TORCH_WARN("torch.ger is deprecated and will be removed in a future PyTorch release. "
             "Use torch.outer instead.");
  return at::outer_out(result, self, vec2);
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  return self.outer(vec2);
}

Tensor& inner_out(Tensor& out, const Tensor& self, const Tensor& other) {
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
      self.size(-1) == other.size(-1),
      "inner() the last dimension must match on both input tensors but got shapes ",
      self.sizes(),
      " and ",
      other.sizes());

  return at::tensordot(self, other, -1, -1);
}

Tensor& outer_out(Tensor &result, const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  // torch.outer is implemented as a composite op using reshape and mul
  at::mul_out(result, self.reshape({self.size(0), 1}), vec2);
  return result;
}

Tensor outer(const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "outer");
  check_1d(vec2, "vec2", "outer");

  return self.reshape({self.size(0), 1}) * vec2;
}

static void addmm_impl_cpu_(
    Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha) {
  TORCH_INTERNAL_ASSERT(self.dim() == 2 && m1.dim() == 2 && m2.dim() == 2);

  // Array access is faster than .size(n) and .stride(n)
  const auto self_sizes = self.sizes();
  auto m1_strides = m1.strides();
  auto m1_sizes = m1.sizes();
  auto m2_strides = m2.strides();
  auto m2_sizes = m2.sizes();

  TORCH_CHECK(
      m1_sizes[1] == m2_sizes[0], "mat1 and mat2 shapes cannot be multiplied (",
      m1_sizes[0], "x", m1_sizes[1], " and ", m2_sizes[0], "x", m2_sizes[1], ")");

  TORCH_CHECK(
      self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
      "input shape is incompatible with matrix multiplication (",
      m1_sizes[0], "x", m1_sizes[1], " @ ", m2_sizes[0], "x", m2_sizes[1], " != ",
      self_sizes[0], "x", self_sizes[1], ")");

  native::resize_(result, self_sizes);
  const auto result_strides = result.strides();
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
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
    c = result;
  } else if (result_strides[1] == 1 &&
             (result_sizes[0] == 1 || result_strides[0] >= std::max(int64_t{1}, result_sizes[1]))) {
    std::swap(m1, m2);
    std::swap(m1_sizes, m2_sizes);
    std::swap(m1_strides, m2_strides);
    transpose_c = true;
    c = result;
  } else {
    transpose_c = false;
    // make c FORTRAN contiguous
    c = result.transpose(0, 1).contiguous().transpose_(0, 1);
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
    a = m1;
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
    b = m2;
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

  // Apply BLAS routine
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16,
      result.scalar_type(), "addmm_impl_cpu_",
      [&]{
        at::native::cpublas::gemm(
            transpose_a ? cpublas::Transpose : cpublas::NoTranspose,
            transpose_b ? cpublas::Transpose : cpublas::NoTranspose,
            m, n, k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(), lda,
            b.data_ptr<scalar_t>(), ldb,
            beta.to<scalar_t>(),
            c.data_ptr<scalar_t>(), ldc);
      });

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
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    result.addmm_(batch1[batch], batch2[batch], adjusted_beta, alpha);
    adjusted_beta = 1; // accumulate output once
  }
}

Tensor& addbmm_out(Tensor& result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor b_self = std::get<0>(expand_size(self, {batch1.size(1), batch2.size(2)}, "addbmm_out"));
  {
    at::NoNamesGuard guard;
    addbmm_impl_(result, b_self, batch1, batch2, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmm(result, batch1, batch2, self);
  return result;
}

Tensor &addbmm_(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return native::addbmm_out(self, self, batch1, batch2, beta, alpha);
}

Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return native::addbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& addmm_cpu_out(Tensor &result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
  Tensor b_self = std::get<0>(expand_size(self, {mat1.sizes()[0], mat2.sizes()[1]}, "addmm_out"));
  {
    at::NoNamesGuard guard;
    addmm_impl_cpu_(result, b_self, mat1, mat2, beta, alpha);
  }
  at::namedinference::propagate_names_for_addmm(result, mat1, mat2, self);
  return result;
}

Tensor addmm_cpu(const Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return addmm_cpu_out(result, self, mat1, mat2, beta, alpha);
}

Tensor &addmm_cpu_(Tensor& self, const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) {
  return addmm_cpu_out(self, self, mat1, mat2, beta, alpha);
}

Tensor& mm_cpu_out(const Tensor & self, const Tensor & mat2, Tensor & result) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  native::resize_(result, {self.sizes()[0], mat2.sizes()[1]});
  return addmm_cpu_out(result, result, self, mat2, 0, 1);
}

Tensor mm_cpu(const Tensor & self, const Tensor & mat2) {
  TORCH_CHECK(self.dim() == 2, "self must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  Tensor result = at::empty({self.sizes()[0], mat2.sizes()[1]}, self.options());
  return addmm_cpu_out(result, result, self, mat2, 0, 1);
}

template <typename scalar_t, bool is_bmm>
inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, const Scalar& beta_, const Scalar& alpha_) {
  int64_t bs = result.size(0);
  int64_t is = result.size(1);
  int64_t js = result.size(2);
  int64_t ks = self.size(2);

  scalar_t alpha = alpha_.to<scalar_t>();
  scalar_t beta = beta_.to<scalar_t>();

  auto r0 = result.accessor<scalar_t, 3>();
  auto s0 = self.accessor<scalar_t, 3>();
  auto m0 = mat2.accessor<scalar_t, 3>();

  int64_t grain_size = std::min(internal::GRAIN_SIZE / (is * js * ks), (int64_t)1);
  parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
      for (int64_t b = b_begin; b < b_end; b++) {
        auto r1 = r0[b];
        auto s1 = s0[b];
        auto m1 = m0[b];
        for (int64_t i = 0; i < is; i++) {
          auto r2 = r1[i];
          auto s2 = s1[i];
          for (int64_t j = 0; j < js; j++) {
            scalar_t &r = r2[j];
            if (is_bmm) {
              r = 0;
              for (int64_t k = 0; k < ks; k++) {
                r += s2[k] * m1[k][j];
              }
            } else {
              r *= beta;
              for (int64_t k = 0; k < ks; k++) {
                r += alpha * s2[k] * m1[k][j];
              }
            }
          }
        }
      }
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

static inline Tensor& bmm_out_or_baddbmm_(Tensor& self_or_result, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, bool is_bmm_out) {
  // is_bmm_out: true for bmm_out, false for baddbmm_
  // self_or_result is "self" for baddbmm_ and "result" for bmm_out
  CheckedFrom c = (is_bmm_out ? "bmm" : "baddbmm");

  auto checkOnCPU = [](const Tensor& t, CheckedFrom c) {
    TORCH_CHECK(
        !t.is_cuda(),
        "Expect tensor to have CPU backend, but got tensor with ",
        toString(t.options().backend()),
        " Backend (while checking arguments for ",
        c);
  };

  checkOnCPU(self_or_result, c);
  checkOnCPU(batch1, c);
  checkOnCPU(batch2, c);

  checkDim(c, batch1, "batch1", /* pos */ 1, /* dim */ 3);
  checkDim(c, batch2, "batch2", /* pos */ 2, /* dim */ 3);

  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();

  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];

  TORCH_CHECK(batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size);

  if (is_bmm_out) {
    self_or_result.resize_({bs, res_rows, res_cols});
  } else {
    const auto self_sizes = self_or_result.sizes();
    TORCH_CHECK(self_sizes[0] == bs && self_sizes[1] == res_rows && self_sizes[2] == res_cols);
  }

  // handle pathological cases that blas may not like
  if (self_or_result.numel() == 0) {
    return self_or_result;
  } else if (contraction_size == 0) {
    if (is_bmm_out || (beta.to<c10::complex<double>>() == 0.0)) {
      return self_or_result.zero_();
    } else {
      return self_or_result.mul_(beta);
    }
  }

  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
    const auto sizes = t.sizes();
    const auto strides = t.strides();
    return (strides[2] == 1 && strides[1] >= sizes[2])
            || (strides[1] == 1 && strides[2] >= sizes[1]);
  };

  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, batch1.scalar_type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, batch1.scalar_type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else if (at::hasMKL() && ((
            self_or_result.scalar_type() != kHalf &&
            self_or_result.scalar_type() != kBFloat16 &&
            at::native::is_floating_point(self_or_result)) ||
            at::native::is_complex(self_or_result))
            && batch_items_contiguous_or_transposed(batch1)
            && batch_items_contiguous_or_transposed(batch2)
            && self_or_result.is_contiguous()) {
    at::native::_baddbmm_mkl_(self_or_result, batch1, batch2, beta, alpha);
  } else { // split along batch dimension
    if (is_bmm_out) {
      for (int64_t b = 0; b < bs; b++) {
        auto r = self_or_result.select(0, b);
        native::mm_cpu_out(batch1.select(0, b), batch2.select(0, b), r);
      }
    } else {
      for (int64_t b = 0; b < bs; b++) {
        self_or_result.select(0, b).addmm_(batch1.select(0, b), batch2.select(0, b), beta, alpha);
      }
    }
  }
  return self_or_result;
}


Tensor baddbmm_cpu(const Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  Tensor result = at::empty({0}, self.options());
  return at::native::baddbmm_out_cpu(self, batch1, batch2, beta, alpha, result);
}

Tensor& baddbmm_out_cpu(const Tensor& self_, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha, Tensor &result) {
  Tensor self;
  std::tie(self) = expand_size(self_, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  result.resize_(self.sizes());
  result.copy_(self);
  return at::native::baddbmm__cpu(result, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cpu(Tensor& self, const Tensor& batch1, const Tensor& batch2, const Scalar& beta, const Scalar& alpha) {
  return bmm_out_or_baddbmm_(self, batch1, batch2, beta, alpha, false);
}

Tensor bmm_cpu(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({0}, self.options());
  return at::native::bmm_out_cpu(self, mat2, result);
}

Tensor& bmm_out_cpu(const Tensor& batch1, const Tensor& batch2, Tensor &result) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  {
  NoNamesGuard guard;
  bmm_out_or_baddbmm_(result, batch1, batch2, beta, alpha, true);
  }
  namedinference::propagate_names_if_nonempty(
      result,
      namedinference::compute_bmm_outnames(result, batch1, batch2));
  return result;
}

Tensor& dot_out(const Tensor& self, const Tensor& tensor, Tensor& result) {
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match self dtype ", self.scalar_type());
  return result.fill_(self.dot(tensor));
}

Tensor& vdot_out(const Tensor& self, const Tensor& other, Tensor& result) {
  at::native::resize_output(result, {});
  TORCH_CHECK(result.scalar_type() == self.scalar_type(),
           "result dtype ", result.scalar_type(), " does not match self dtype ", self.scalar_type());
  return result.fill_(self.vdot(other));
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are broadcasted (and thus
  must be broadcastable).  For example, if tensor1 is a (j x 1 x n x m) Tensor
  and tensor2 is a (k x m x p) Tensor, the returned tensor will be an (j x k x n x p) Tensor.
*/
Tensor matmul(
    c10::optional<Tensor> out_opt,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  NoNamesGuard guard;
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  auto has_out = out_opt.has_value();
  Tensor out = out_opt.value_or(Tensor());

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::native::dot_out(tensor1, tensor2, out) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (dim_tensor1 >= 3 && (dim_tensor2 == 1 || dim_tensor2 == 2)) {
    // optimization: use mm instead of bmm by folding tensor1's batch into
    // its leading matrix dimension.

    Tensor t2 = dim_tensor2 == 1 ? tensor2.unsqueeze(-1) : tensor2;
    auto size1 = tensor1.sizes();
    auto size2 = t2.sizes();
    std::vector<int64_t> output_size;
    output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
    if (dim_tensor2 > 1) {
      output_size.push_back(size2[dim_tensor2 - 1]);
    }

    // fold the batch into the first dimension
    Tensor t1 = tensor1.contiguous().view({-1, size1[size1.size() - 1]});
    Tensor output = has_out ? at::_unsafe_view(at::mm_out(out, t1, t2), output_size)
                            : at::_unsafe_view(t1.mm(t2), output_size);
    return has_out ? out.set_(output) : output;
  } else if ((dim_tensor1 == 1 || dim_tensor1 == 2) && dim_tensor2 >= 3) {
    // optimization: transpose the inner dimensions of the arguments, call
    // matmul on the swapped arguments, then transpose the inner dimensions
    // of the result.
    const int64_t n = dim_tensor1 == 2 ? tensor1.size(-2) : 1;
    const int64_t m = tensor1.size(-1);
    const int64_t p = tensor2.size(-1);

    const Tensor t2_T = tensor2.transpose(-1, -2);
    const Tensor t1_T = dim_tensor1 == 2 ? tensor1.t() : tensor1.reshape({n, m}).t();
    const Tensor res_T = matmul(out_opt, t2_T, t1_T);

    if (dim_tensor1 == 2) {
      Tensor res = res_T.transpose(-1, -2).contiguous();
      return has_out ? out.set_(res) : res;
    }
    else {
      std::vector<int64_t> shape = tensor2.sizes().slice(0, dim_tensor2 - 2).vec();
      shape.push_back(p);

      Tensor res = res_T.reshape(shape).contiguous();
      return has_out ? out.set_(res) : res;
    }
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntArrayRef batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntArrayRef batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    const int64_t expand_batch_product =
        c10::multiply_integers(expand_batch_portion);

    std::vector<int64_t> tensor1_bmm_view({expand_batch_product});
    tensor1_bmm_view.insert(tensor1_bmm_view.end(), {n, m1});

    std::vector<int64_t> tensor2_bmm_view({expand_batch_product});
    tensor2_bmm_view.insert(tensor2_bmm_view.end(), {m2, p});

    // flatten expanded batches
    Tensor tensor1_expanded = tensor1.expand(tensor1_expand_size).contiguous().view(tensor1_bmm_view);
    Tensor tensor2_expanded = tensor2.expand(tensor2_expand_size).contiguous().view(tensor2_bmm_view);

    // reshape batches back into result
    std::vector<int64_t> output_shape(expand_batch_portion);
    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    Tensor output = has_out ? at::_unsafe_view(at::bmm_out(out, tensor1_expanded, tensor2_expanded), output_shape)
                            : at::_unsafe_view(tensor1_expanded.bmm(tensor2_expanded), output_shape);

    return has_out ? out.set_(output) : output;
  }

 AT_ERROR("both arguments to matmul need to be at least 1D, but they are ",
          dim_tensor1, "D and ", dim_tensor2, "D");
}

Tensor matmul(const Tensor & tensor1, const Tensor & tensor2) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  auto result = at::native::matmul(c10::nullopt, tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& matmul_out(const Tensor & tensor1, const Tensor & tensor2, Tensor &result) {
  auto maybe_outnames = namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::native::matmul(c10::optional<Tensor>(result), tensor1, tensor2);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
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
    at::native::matmul(
      buffer.select(0, 2), // out for a^2
      buffer.select(0, 1),
      buffer.select(0, 1)
    );
  }

  // fill a^3
  if (3 <= num_matrices - 1) {
    at::native::matmul(
      buffer.select(0, 3), // out for a^3
      buffer.select(0, 1),
      buffer.select(0, 2)
    );
  }

  // fill a^6
  if (4 <= num_matrices - 1) {
    at::native::matmul(
      buffer.select(0, 4),
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
    c10::toValueType(in.scalar_type())).unsqueeze(0);
  return _move_memory_if_cuda_input(tensor, in);
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

  at::native::matmul(
    // output for A^2 * (I / 2 + A / 6 + A^2 / 24)
    As.select(0, 3),
    // contains A^2
    As.select(0, 2),
    // computes (I / 2 + A / 6 + A^2 / 24)
    at::native::_compute_linear_combination(
      As.narrow(0, 0, 3),
      _blob_to_Tensor<scalar_t>({1 / 2.0, 1 / 6.0, 1 / 24.0}, A)
    )
  );

  // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
  return at::native::_compute_linear_combination(
    As, _blob_to_Tensor<scalar_t>({1.0, 1.0, 0.0, 1.0}, A)
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

  // A4 =  A2 * (x1 * A + x2 * A2)
  at::native::matmul(
    // output for A4
    As.select(0, 3),
    // As.select(0, 2) = A^2
    As.select(0, 2),
    at::native::_compute_linear_combination(
      // extract {A, A^2} from As
      As.narrow(0, 1, 2),
      _blob_to_Tensor<scalar_t>({x1, x2}, A)
    )
  );

  // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
  at::native::matmul(
    // output for A8
    As.select(0, 4),
    // x3 * A2 + A4
    at::native::_compute_linear_combination(
      As.narrow(0, 2, 2),
      _blob_to_Tensor<scalar_t>({x3, 1.0}, A)
    ),
    at::native::_compute_linear_combination(
      As.narrow(0, 0, 4),
      _blob_to_Tensor<scalar_t>({x4, x5, x6, x7}, A)
    )
  );

  // return I + A + y2 * A2 + A8;
  return at::native::_compute_linear_combination(
    As,
    _blob_to_Tensor<scalar_t>({1.0, 1.0, y2, 0.0, 1.0}, A)
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
    c10::toValueType(A.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = at::native::_compute_linear_combination(As, bs);

  // compute A6
  Bs.select(0, 2).add_(at::native::matmul(
    // tmp buffer for this matrix product
    As.select(0, 0),
    Bs.select(0, 3),
    Bs.select(0, 3)
  ));

  return Bs.select(0,0).add_(at::native::matmul(
    // tmp buffer for this matrix product
    As.select(0, 0),
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
    c10::toValueType(A.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, A);

  auto As = _allocate_buffer(A, num_prods);
  _fill_matrix_powers(As, A, num_prods);

  auto Bs = at::native::_compute_linear_combination(As, bs);

  // compute A9
  Bs.select(0, 3).add_(at::native::matmul(
    // tmp buffer for this matrix product
    As.select(0, 0),
    Bs.select(0, 0),
    Bs.select(0, 4))
  );

  return Bs.select(0, 1).add_(at::native::matmul(
    // tmp buffer for this matrix product
    As.select(0, 0),
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
  for (int64_t i = 0; i < mexp_scaled.size(0); ++i) {
    auto s_val = s_cpu.select(0, i).template item<int64_t>();
    auto mexp = mexp_scaled.select(0, i);
    for (int64_t p = 0; p < s_val; ++p) {
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

// Based on:
//
// Mathias, Roy.
// A Chain Rule for Matrix Functions and Applications.
// SIAM J. Matrix Anal. Appl. 17 (1996): 610-620.
//
template <typename func_t>
Tensor backward_analytic_function_of_a_matrix(
    const Tensor& self, const Tensor& grad,
    const func_t& function_of_a_matrix
  ) {
  auto self_transposed = self.transpose(-2, -1).conj();
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

};

// Computes the matrix exponential for a given batch of squared matrices.
// The implementaion is based on:
//
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
// Mathematics 2019, 7, 1174.
//
Tensor matrix_exp(const Tensor& a) {
  TORCH_CHECK(a.dim() >= 2
          && (at::isFloatingType(a.scalar_type())
           || at::isComplexType(a.scalar_type())),
              "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
              "of floating or complex types with dim at least 2");
  TORCH_CHECK(a.size(-1) == a.size(-2),
              "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
              "of squared matrices");

  NoTF32Guard disable_tf32;

  if (a.size(-1) == 1) {
    return a.exp();
  }

  return mexp(a);
}

Tensor matrix_exp_backward(const Tensor& self, const Tensor& grad) {
  NoTF32Guard disable_tf32;
  return backward_analytic_function_of_a_matrix(
    self, grad,
    [](const Tensor& a) {
      return a.matrix_exp();
    }
  );
}

Tensor matrix_power(const Tensor& a, int64_t n) {
  TORCH_CHECK(a.dim() >= 2 && (at::isFloatingType(a.scalar_type()) || at::isComplexType(a.scalar_type())),
              "matrix_power(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
              "of floating types with dim at least 2");
  if (n == 0) {
    return a.clone(at::MemoryFormat::Contiguous).copy_(at::eye(a.size(-2), a.options()).expand_as(a));
  } else if (n < 0) {
    Tensor a_ = at::inverse(a);
    n *= -1;
    return at::native::matrix_power(a_, n);
  } else if (n == 1) {
    return a.clone(at::MemoryFormat::Contiguous);
  } else if (n == 2) {
    return at::native::matmul(a, a);
  } else if (n == 3) {
    return at::native::matmul(at::native::matmul(a, a), a);
  }

  // This is a binary decomposition of n.
  // Moving from the least significant bit to the most significant bit
  // This is done to reduce the number of matrix multiplications
  // by raising the input matrix in powers of 2
  // The total number of matrix multiplications are
  // number of bits + number of bits that equal 1 ~ O(log n)
  // instead of O(n)
  Tensor result, z;
  int64_t r;
  while (n > 0) {
    z = (!z.defined()) ? a.clone(at::MemoryFormat::Contiguous) : at::native::matmul(z, z);
    r = n % 2;
    n = n / 2;
    if (r == 1) {
      result = (!result.defined()) ? z.clone(at::MemoryFormat::Contiguous) : at::native::matmul(result, z);
    }
  }
  return result;
}

Tensor frobenius_norm(const Tensor& self) {
  return at::norm(self);
}

Tensor frobenius_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  // NOTE: As frobenius_norm_out is currently implemented, it will always produce a
  //    strided tensor result, even if the input is sparse.
  auto options = self.options().layout(c10::Layout::Strided).dtype(toValueType(self.scalar_type()));
  Tensor result = at::empty({0}, options);
  return at::native::frobenius_norm_out(result, self, dim, keepdim);
}

Tensor &frobenius_norm_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  TORCH_CHECK(
      dim.size() <= 2,
      "Expected at most 2 dimensions, but got ",
      dim.size(),
      " dimensions instead.");
  Tensor result_;
  if (dim.size() == 1 || dim.size() == 0) {
    result_ = at::norm(self, 2, dim, keepdim);
  } else {
    auto dim_ = dim.vec();
    maybe_wrap_dims(dim_, self.dim());
    TORCH_CHECK(dim_[0] != dim_[1], "Expected dims to be different, got ", dim, " instead");
    if (self.is_complex()){
      result_ = at::sqrt(at::sum(at::real(self.conj() * self), dim_, keepdim));
    } else {
      result_ = at::sqrt(at::sum((self * self), dim_, keepdim));
    }
  }
  // NOTE: It would be better to avoid resize and copy by using norm_out and sqrt_out above.
  //    However, norm_out and sqrt_out do not support automatic differentiation.
  //    More details here: https://github.com/pytorch/pytorch/pull/44095#discussion_r486673947
  resize_output(result, result_.sizes());
  result.copy_(result_);
  return result;
}

Tensor nuclear_norm(const Tensor& self, bool keepdim) {
  TORCH_CHECK(
      self.dim() == 2,
      "Expected a tensor with 2 dimensions, but got a tensor with ",
      self.dim(), " dimension", self.dim()==1 ? "" : "s", " instead.");
  return at::native::nuclear_norm(self, IntArrayRef({0, 1}), keepdim);
}

Tensor &nuclear_norm_out(Tensor& result, const Tensor& self, bool keepdim) {
  TORCH_CHECK(
      self.dim() == 2,
      "Expected a tensor with 2 dimensions, but got a tensor with ",
      self.dim(), " dimension", self.dim()==1 ? "" : "s", " instead.");
  return at::native::nuclear_norm_out(result, self, IntArrayRef({0, 1}), keepdim);
}

Tensor nuclear_norm(const Tensor& self, IntArrayRef dim, bool keepdim) {
  Tensor result = at::empty({0}, self.options().dtype(toValueType(self.scalar_type())));
  return at::native::nuclear_norm_out(result, self, dim, keepdim);
}

Tensor& nuclear_norm_out(Tensor& result, const Tensor& self, IntArrayRef dim, bool keepdim) {
  TORCH_CHECK(dim.size() == 2, "nuclear norm requires a 'dim' argument of size 2");
  auto dim_ = dim.vec();
  maybe_wrap_dims(dim_, self.dim());

  auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], self.dim());
  Tensor p = self.permute(permutation);
  // NOTE: U and V are computed only if gradmode is enabled, since the backward for nuclear
  //       norm uses svd_backward, which requires them.
  Tensor result_ = at::sum(std::get<1>(at::svd(p, /*some=*/true,
                  /*compute_uv=*/at::GradMode::is_enabled() && self.requires_grad())), -1, keepdim);
  if (keepdim) {
    result_.unsqueeze_(-1);
    auto permutation_reverse = create_reverse_permutation(permutation);
    result_ = result_.permute(permutation_reverse);
  }
  resize_output(result, result_.sizes());
  result.copy_(result_);
  return result;
}

// Creates a vector of length ndim with values equal to its indices
// (e.g. [0, 1, 2, ..., ndim-1])
static std::vector<int64_t> make_dim_list(int64_t ndim) {
  std::vector<int64_t> dim_list(ndim);
  for (int64_t ind = 0; ind < ndim; ind++) {
    dim_list[ind] = ind;
  }
  return dim_list;
}

// Checks for valid arguments to linalg_norm when type(ord) == str
static void check_str_ord_valid(const std::string& str_ord, optional<IntArrayRef> opt_dim, int64_t ndim) {
  TORCH_CHECK((str_ord == "nuc") || (str_ord == "fro"), "Invalid norm order: ", str_ord);
  bool dims_valid = (ndim == 2 && !opt_dim.has_value()) || (opt_dim.has_value() && opt_dim.value().size() == 2);
  TORCH_CHECK(dims_valid, "order \"", str_ord,
    "\" can only be used if either len(dim) == 2 or (self.dim() == 2 and dim is None)");
}

// Performs vector norm for ord = +/-infinity, and the second dimension reduction
// for matrix norms.
static Tensor _norm_min_max(Tensor& self, double ord, int64_t dim, bool keepdim) {
  Tensor result;
  if (self.numel() == 0 && self.sizes()[dim] > 0) {
    // This special case is needed in matrix norm for tensors with 3 or more dims,
    // or in vector norm for order inf and -inf for tesnsors with 2 or more dims.
    // When the sizes of the dims to be reduced are greater than 0 but another dim
    // in the tensor is size 0 (thus numel == 0), we must either flatten or resize
    // the second reduction dim to 1, to avoid calling min/max, which would throw
    // an error.
    if (self.sizes()[dim] != 1) {
      auto new_sizes = self.sizes().vec();
      new_sizes[dim] = 1;
      self.resize_(new_sizes);
    }
    result = keepdim ? self : self.flatten(dim);
  } else {
    if (ord > 0) {
      result = std::get<0>(self.max(dim, keepdim));
    } else {
      result = std::get<0>(self.min(dim, keepdim));
    }
  }
  return result;
}

// Performs matrix norm
static Tensor& _linalg_norm_matrix_out(Tensor& result, const Tensor &self, const optional<Scalar>& opt_ord,
                               IntArrayRef dim, bool keepdim, optional<ScalarType> opt_dtype) {
  Tensor result_;
  auto ord = opt_ord.value_or(2.0).toDouble();
  TORCH_CHECK(self.device().is_cpu() || self.is_cuda(),
              "matrix norm only supports CPU AND CUDA device type, got: ", self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "matrix norm only supports strided layout, got: ", self.layout());

  TORCH_CHECK(dim.size() == 2, "_linalg_norm_matrix: 'dim' must either specify 2 dimensions. ",
    "Got 'dim' specifying ", dim.size(), " dims");
  auto dim_ = dim.vec();
  maybe_wrap_dims(dim_, self.dim());
  TORCH_CHECK(dim_[0] != dim_[1],
    "Expected dims to be different, got (", dim[0], ", ", dim[1], ") instead");

  ScalarType scalarType = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
  TORCH_CHECK(
      at::isFloatingType(scalarType) || at::isComplexType(scalarType),
      "Can only calculate the mean of floating and complex types. Got ",
      toString(scalarType), " instead.");

  Tensor self_;
  if (opt_dtype.has_value()) {
    self_ = self.to(scalarType);
  } else {
    self_ = self;
  }

  if (std::abs(ord) == 2) {
    // Need to shift the reduction dims to the back, because at::svd will only operate on
    // the last 2 dimensions
    auto permutation = create_dim_backshift_permutation(dim_[0], dim_[1], self.dim());
    auto permutation_reverse = create_reverse_permutation(permutation);

    result_ = std::get<1>(self_.permute(permutation).svd()).abs();
    result_ = _norm_min_max(result_, ord, result_.dim() - 1, keepdim);

    if (keepdim) {
      result_.unsqueeze_(-1);
      result_ = result_.permute(permutation_reverse);
    }
  } else {
    // abs(p) == infinity and abs(p) == 1 will perform identical reductions, except
    // that the order of the two dims is swapped. So we can swap the dims if
    // abs(p) == infinity to simplify the rest of the operation's logic.
    if (std::abs(ord) == INFINITY) {
      std::swap(dim_[0], dim_[1]);
    }
    // If the dim of the second reduction is greater than that of the first reduction
    // and we are not keeping the dims, then the fact that the output of the first
    // reduction will have one fewer dimension means that the second reduction dim
    // will be off by one, so we need to correct that.
    if ((dim_[1] > dim_[0]) && !keepdim) {
      dim_[1]--;
    }
    if (std::abs(ord) == 1 || std::abs(ord) == INFINITY) {
      result_ = self_.abs().sum(dim_[0], keepdim);
      result_ = _norm_min_max(result_, ord, dim_[1], keepdim);
    } else {
      TORCH_CHECK(false, "Order ", ord, " not supported for matrix norm");
    }
  }
  resize_output(result, result_.sizes());
  result.copy_(result_);
  return result;
}

static Tensor& linalg_norm_out_impl(Tensor& result, const Tensor& self, const optional<Scalar>& opt_num_ord, optional<std::string> opt_str_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  // Callers must give the ord argument as either a number, a string, or neither.
  // Since the user-facing API has no direct control over how this function is called, this is an internal assert.
  TORCH_INTERNAL_ASSERT(!(opt_num_ord.has_value() && opt_str_ord.has_value()));
  if (opt_dtype.has_value()) {
    auto dtype = opt_dtype.value();
    TORCH_CHECK(dtype == result.scalar_type(), "provided dtype must match dtype of result, but got",
      "dtype = ", dtype, ", out.dtype = ", result.scalar_type());
  }
  int64_t ndim = self.dim();
  if (opt_str_ord.has_value()) {
    // 'ord' is string
    auto str_ord = opt_str_ord.value();
    check_str_ord_valid(str_ord, opt_dim, ndim);
    Tensor self_ = opt_dtype.has_value() ? self.to(opt_dtype.value()) : self;
    if (str_ord == "fro") {
      at::frobenius_norm_out(result, self_, opt_dim.value_or(IntArrayRef({0, 1})), keepdim);
    } else if (str_ord == "nuc") {
      if (opt_dim.has_value()) {
        at::nuclear_norm_out(result, self_, opt_dim.value(), keepdim);
      } else {
        at::nuclear_norm_out(result, self_, keepdim);
      }
    }
  } else {
    // 'ord' is int or None
    std::vector<int64_t> dim_ = opt_dim.has_value() ? opt_dim.value().vec() : make_dim_list(ndim);
    if (!opt_num_ord.has_value() || dim_.size() == 1) {
      Tensor result_ = at::linalg_vector_norm(self, opt_num_ord, opt_dim, keepdim, opt_dtype);
      // TODO: Resize and copy should be avoided with
      //       https://github.com/pytorch/pytorch/issues/52712
      resize_output(result, result_.sizes());
      result.copy_(result_);
    } else if (dim_.size() == 2) {
      _linalg_norm_matrix_out(result, self, opt_num_ord.value(), dim_, keepdim, opt_dtype);
    } else {
      TORCH_CHECK(false, "'dim' must specify 1 or 2 dimensions when order is numerical and input is "
        "not 1-D or 2-D");
    }
  }
  return result;
}

static Tensor& linalg_vector_norm_impl(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  // Casting a large integer to a double will introduce some error, but for
  // practical purposes, it won't matter since a large order will usually
  // give an infinite result
  auto ord = opt_ord.value_or(2).toDouble();

  TORCH_CHECK(self.device().type() == DeviceType::CPU || self.device().type() == DeviceType::CUDA,
              "linalg.vector_norm only supports CPU and CUDA device types, but got: ",
              self.device().type());
  TORCH_CHECK(self.layout() == Layout::Strided,
              "linalg.vector_norm only supports strided layout, but got: ", self.layout());

  if (opt_dtype.has_value() && isComplexType(self.scalar_type())) {
    TORCH_CHECK(isComplexType(opt_dtype.value()),
      "linalg.vector_norm expected complex 'dtype', since input is complex, ",
      "but got ", opt_dtype.value());
  }

  ScalarType in_dtype = opt_dtype.value_or(self.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(in_dtype) || at::isComplexType(in_dtype),
      "linalg.vector_norm only supports floating point and complex dtypes, but got: ",
      toString(in_dtype));

  IntArrayRef dim = opt_dim.value_or(IntArrayRef{});

  if (self.numel() == 0) {
    // TODO: The question about how to handle negative orders when the input
    // is empty has not been settled yet. For now, we raise an error. Issue:
    // https://github.com/pytorch/pytorch/issues/52783
    TORCH_CHECK(ord >= 0,
      "linalg.vector_norm of negative order cannot be performed on an empty tensor");

    // For NumPy compatibility, we can only perform order infinity reduction
    // (max/min) on a tensor with zero elements if the dimensions to reduce are
    // nonzero. Otherwise, throw an error.
    if (ord == INFINITY) {
      bool has_identity = true;

      if (dim.size() == 0) {
        has_identity = false;
      } else {
        for (int64_t dim_num : dim) {
          if (self.size(dim_num) == 0) {
            has_identity = false;
            break;
          }
        }
      }
      TORCH_CHECK(has_identity,
        "linalg.vector_norm cannot compute the infinity norm on an empty ",
        "dimension because the operation does not have an identity");
    }
  }
  Tensor self_;
  if (self.device().type() == c10::kCPU && isComplexType(self.scalar_type()) && std::abs(ord) == INFINITY) {
    // TODO: This at::abs() call is used so that the at::abs() call in the
    // backward function produces an identical result for complex inputs.
    // However, it would be ideal if we could incorporate this into
    // linalg_vector_norm_stub. See issue:
    // https://github.com/pytorch/pytorch/issues/52648
    self_ = self.to(in_dtype).abs();
    in_dtype = toValueType(in_dtype);
  } else {
    self_ = self;
  }
  ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
  TORCH_CHECK(!result.defined() || out_dtype == result.scalar_type(),
    "linalg.vector_norm expected out tensor dtype ", out_dtype,
    " but got: ", result.scalar_type());
  auto iter = make_reduction("vector_norm", result, self_, dim, keepdim, in_dtype, out_dtype);
  linalg_vector_norm_stub(iter.device_type(), iter, ord);
  return result;
}

Tensor linalg_vector_norm(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  ScalarType out_dtype = opt_dtype.value_or(toValueType(self.scalar_type()));
  Tensor result = create_reduction_result(self, opt_dim.value_or(IntArrayRef{}), keepdim, out_dtype);
  return at::native::linalg_vector_norm_impl(self, opt_ord, opt_dim, keepdim, opt_dtype, result);
}

Tensor& linalg_vector_norm_out(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype, Tensor& result) {
  return at::native::linalg_vector_norm_impl(self, opt_ord, opt_dim, keepdim, opt_dtype, result);
}

// Numerical or None norms
Tensor linalg_norm(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto options = TensorOptions().dtype(opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type())).device(self.device());
  Tensor result = at::empty({0}, options);
  return at::native::linalg_norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

// Frobenius and nuclear norms
Tensor linalg_norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  auto options = TensorOptions().dtype(opt_dtype.has_value() ? opt_dtype.value() : toValueType(self.scalar_type())).device(self.device());
  Tensor result = at::empty({0}, options);
  return at::native::linalg_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

// Numerical or None norms
Tensor& linalg_norm_out(Tensor& result, const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return linalg_norm_out_impl(result, self, opt_ord, c10::nullopt, opt_dim, keepdim, opt_dtype);
}

// Frobenius and nuclear norms
Tensor& linalg_norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return linalg_norm_out_impl(result, self, c10::nullopt, ord, opt_dim, keepdim, opt_dtype);
}

Tensor _linalg_cond_exception_helper(const Tensor& self) {
  // For batched input if at least one matrix in the batch is not invertible,
  // we can't get the result for all other (possibly) invertible matrices in the batch without an explicit for loop.
  // This should change when at::inverse works with silent errors
  if (self.dim() > 2) {
    TORCH_CHECK(false,
      "One or more matrices in the batch was not invertible! "
      "linalg_cond does not support yet this case.");
  }
  auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
  TensorOptions options = self.options().dtype(toValueType(self.scalar_type()));
  Tensor result = at::full(result_shape, INFINITY, options);
  return result;
}

// This function helps to dispatch norm computations depending on 'ord' of variant type
Tensor _linalg_cond_helper(const Tensor& self, c10::variant<Scalar, std::string> ord_variant) {
  // Ignore errors if not invertible, result is INFINITY in this case
  // Currently checking for error in at::inverse causes cross-device data movement
  // For batched input if at least one matrix in the batch is not invertible,
  // then the result for all other (possibly) invertible matrices will be infinity as well
  // since there is currently no way to use at::inverse with silent errors
  Tensor self_inverse;
  try {
    self_inverse = at::inverse(self);
  } catch (const std::exception& e) {
    if (strstr(e.what(), "singular")) {
      return _linalg_cond_exception_helper(self);
    } else {
      TORCH_CHECK(false, "linalg_cond got an unexpected error:\n", e.what());
    }
  }
  std::array<int64_t, 2> dim_arr = {-2, -1};
  optional<IntArrayRef> dim = IntArrayRef(dim_arr);

  return c10::visit([&](auto&& ord) {
    Tensor norm_self = at::linalg_norm(self, ord, dim);
    Tensor norm_inverse = at::linalg_norm(self_inverse, ord, dim);
    Tensor result = norm_self * norm_inverse;
    return result;
  }, ord_variant);
}

// Return zero for each matrix in the batch
Tensor _linalg_cond_empty_matrix(const Tensor& self, c10::ScalarType dtype) {
  auto result_shape = IntArrayRef(self.sizes().cbegin(), self.sizes().cend()-2);
  TensorOptions options = self.options().dtype(toValueType(self.scalar_type()));
  return at::zeros(result_shape, options);
}

void _linalg_cond_check_ord(c10::variant<Scalar, std::string> ord_variant) {
  if (ord_variant.index() == 0) {
    Scalar* ord = c10::get_if<Scalar>(&ord_variant);
    double abs_ord = std::abs(ord->toDouble());
    TORCH_CHECK(abs_ord == 2.0 || abs_ord == 1.0 || abs_ord == INFINITY,
      "linalg_cond got an invalid norm type: ", ord->toDouble());
  } else if (ord_variant.index() == 1) {
    std::string* ord = c10::get_if<std::string>(&ord_variant);
    TORCH_CHECK(*ord == "fro" || *ord == "nuc",
      "linalg_cond got an invalid norm type: ", *ord);
  } else {
    TORCH_CHECK(false,
      "linalg_cond: something went wrong while checking the norm type");
  }
}

// Numerical or None norms
Tensor linalg_cond(const Tensor& self, const optional<Scalar>& opt_ord) {
  TORCH_CHECK(self.dim() >= 2, "linalg_cond only supports matrices or batches of matrices, but got a tensor with ",
    self.dim(), " dimensions.");

  // The default case is using 2-norm
  Scalar ord = opt_ord.has_value() ? opt_ord.value() : 2;

  c10::variant<Scalar, std::string> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
  if (self.numel() == 0) {
    auto real_dtype = toValueType(typeMetaToScalarType(self.dtype()));
    return _linalg_cond_empty_matrix(self, real_dtype);
  }

  // If ord == None or ord == ±2
  if (std::abs(ord.toDouble()) == 2.0) {
    auto singular_values = std::get<1>(at::svd(self));
    // singular values are sorted in descending order
    auto s_max = at::narrow(singular_values, /*dim=*/-1, /*start=*/0, /*length=*/1);
    auto s_min = at::narrow(singular_values, /*dim=*/-1, /*start=*/-1, /*length=*/1);
    Tensor result;
    if (ord.toDouble() == -2.0) {
      result = s_min / s_max;
    } else {
      result = s_max / s_min;
    }
    return result;
  }

  // ord == ±1 ord == ±inf
  // since at::inverse is used in the implementation, self has to be a tensor consisting of square matrices
  // the same check as squareCheckInputs(self) but with a slightly more informative error message
  TORCH_CHECK(self.size(-1) == self.size(-2),
              "linalg_cond with ±1 or ±inf norm types only supports square matrices or batches of square matrices "
              "but got ", self.size(-1), " by ", self.size(-2), " matrices");

  return _linalg_cond_helper(self, ord_variant);
}

Tensor& linalg_cond_out(Tensor& result, const Tensor& self, const optional<Scalar>& opt_ord) {
  checkSameDevice("linalg_cond", result, self);
  ScalarType real_dtype = toValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg_cond", result.scalar_type(), real_dtype);

  Tensor result_tmp = at::linalg_cond(self, opt_ord);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

// Frobenius or nuclear norms
Tensor linalg_cond(const Tensor& self, std::string ord) {
  // the same checks as squareCheckInputs(self) but with a slightly more informative error message
  TORCH_CHECK(self.dim() >= 2, "linalg_cond only supports matrices or batches of matrices, but got a tensor with ",
    self.dim(), " dimensions.");
  TORCH_CHECK(self.size(-1) == self.size(-2),
              "linalg_cond with frobenius or nuclear norm types only supports square matrices or batches of square matrices "
              "but got ", self.size(-1), " by ", self.size(-2), " matrices");

  c10::variant<Scalar, std::string> ord_variant = ord;
  _linalg_cond_check_ord(ord_variant);

  // NumPy doesn't define the condition number for 0x0 matrices, we return 0.0 for such input
  if (self.numel() == 0) {
    return _linalg_cond_empty_matrix(self, self.scalar_type());
  }

  return _linalg_cond_helper(self, ord_variant);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_cond_out(Tensor& result, const Tensor& self, std::string ord) {
  checkSameDevice("linalg_cond", result, self);
  ScalarType real_dtype = toValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg_cond", result.scalar_type(), real_dtype);

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
  std::vector<int64_t> shape_ind_end = self.sizes().slice(ind).vec();
  // self[:ind]
  std::vector<int64_t> shape_start_ind = self.sizes().slice(0, ind).vec();

  int64_t prod_ind_end = c10::multiply_integers(shape_ind_end.cbegin(), shape_ind_end.cend());
  int64_t prod_start_ind = c10::multiply_integers(shape_start_ind.cbegin(), shape_start_ind.cend());

  // Check whether the self tensor can be reshaped to the 2D square matrix
  TORCH_CHECK(prod_ind_end == prod_start_ind,
    "Expected self to satisfy the requirement prod(self.shape[ind:]) == prod(self.shape[:ind]), but got ",
    prod_ind_end, " != ", prod_start_ind);

  // Concatenate shape_ind_end and shape_start_ind to form the shape of the result
  // self[ind:] + self[:ind]
  shape_ind_end.insert(shape_ind_end.cend(), shape_start_ind.cbegin(), shape_start_ind.cend());

  // If the reshaped self is not invertible catch this error
  Tensor result;
  try {
    result = at::inverse(self.reshape({prod_ind_end, prod_ind_end}));
  } catch (...) {
    TORCH_CHECK(false, "Failed to invert the input tensor, because it is singular.");
  }

  return result.reshape(shape_ind_end);
}

// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_tensorinv_out(Tensor& result, const Tensor& self, int64_t ind) {
  checkSameDevice("tensorinv", result, self);
  checkLinalgCompatibleDtype("tensorinv", result, self);

  Tensor result_tmp = at::linalg_tensorinv(self, ind);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

Tensor linalg_tensorsolve(const Tensor& self, const Tensor& other, optional<IntArrayRef> dims) {
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
  std::vector<int64_t> result_shape = self_.sizes().slice(other.dim(), ndim - other.dim()).vec();

  int64_t result_product = c10::multiply_integers(result_shape.begin(), result_shape.end());
  int64_t other_product = c10::multiply_integers(other.sizes().begin(), other.sizes().end());

  // Check whether the self tensor can be reshaped to the 2D square matrix
  TORCH_CHECK(result_product == other_product,
    "Expected self to satisfy the requirement prod(self.shape[other.ndim:]) == prod(self.shape[:other.ndim]), but got ",
    result_product, " != ", other_product);

  self_ = self_.reshape({result_product, result_product});

  // 0th output of at::solve is the solution
  // normally `other` would be flattened by at::solve expects 2D input
  Tensor result = std::get<0>(at::solve(other.reshape({other.numel(), 1}), self_));
  return result.reshape(result_shape);
}

Tensor& linalg_tensorsolve_out(Tensor& result, const Tensor& self, const Tensor& other, optional<IntArrayRef> dims) {
  checkSameDevice("tensorsolve", result, self);
  checkLinalgCompatibleDtype("tensorsolve", result, self);

  Tensor result_tmp = at::linalg_tensorsolve(self, other, dims);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

static inline Tensor _chain_matmul_general(TensorList matrices, std::vector<std::vector<int64_t>>& order, int64_t i, int64_t j) {
  if (i == j)
    return matrices[i];
  else
    return at::mm(_chain_matmul_general(matrices, order, i, order[i][j]), _chain_matmul_general(matrices, order, order[i][j] + 1, j));
}

// Why the separate implementation for 3 matrices?
// The logic for three matrices is much faster when done directly
// Requires 1 comparison to 4 comparisons and lesser arithmetic operations
static inline Tensor _chain_matmul_three_matrices(TensorList matrices) {
  int64_t a = matrices[0].size(0);  // This is the first dimension
  int64_t b = matrices[1].size(0);  // This is the common dimension between the first two matrices
  int64_t c = matrices[2].size(0);  // This is the common dimension between the last two matrices
  int64_t d = matrices[2].size(1);  // This is the last dimension

  // The matrices are of size (a x b), (b x c), (c x d)
  // cost_1 is the cost of parenthesizing (a x b) and (b x c) and then combining (c x d)
  // cost_2 is the cost of parenthesizing (b x c) and (c x d) and then combining (a x b)
  int64_t cost_1 = (a * c) * (b + d);
  int64_t cost_2 = (b * d) * (a + c);

  if (cost_1 > cost_2) {
    return at::mm(matrices[0], at::mm(matrices[1], matrices[2]));
  } else {
    return at::mm(at::mm(matrices[0], matrices[1]), matrices[2]);
  }
}

Tensor chain_matmul(TensorList matrices) {
  checkAllSameDim(matrices, 2);

  TORCH_CHECK(matrices.size() > 0, "chain_matmul: Expected one or more matrices");
  if (matrices.size() == 1) {
    return matrices[0];
  } else if (matrices.size() == 2) {
    return at::mm(matrices[0], matrices[1]);
  } else if (matrices.size() == 3) {
    return _chain_matmul_three_matrices(matrices);
  } else {

    // Following the algorithm in Chapter 15.2 : Introduction to Algorithms, Cormen et al.
    // Minor modifications have been made to accommodate zero-indexing
    auto n = matrices.size();

    // Dim vector - the length of which is n + 1. Note that for matrix multiplication, there
    // needs to a common dimension between the multiplicands, hence for n matrices, there are
    // n + 1 values. The values p_{i} and p_{i + 1} correspond to the dimensions of matrix i in
    // the chain (zero-indexed)
    std::vector<int64_t> p;
    p.push_back(matrices[0].size(0));
    for (size_t i = 0; i < n; i++) {
      p.push_back(matrices[i].size(1));
    }

    // Cost matrix - an element m[i, j] of this matrix corresponds to the minimum cost of
    // parenthesizing matrices A_{i} to A_{j}. By this definition m[i, i] = 0 for all i
    // m[i, j] is filled using the substructure property of the algorithm, meaning:
    // m[i, j] = min_{i <= k < j} m[i, k] + m[k, j] + p_{i-1}p_{k}p_{j}
    std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

    // Auxiliary table for constructing the order
    // s[i, j] stores the index k at which the optimal split is obtained
    std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

    // j and q are used repetitively in the algorithm below
    int64_t j, q;

    for (int64_t l = 1; l < n; l++) {
      for (int64_t i = 0; i < n - l; i++) {
        j = i + l;
        m[i][j] = std::numeric_limits<int64_t>::max();
        for (int64_t k = i; k < j; k++) {
          q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
          if (q < m[i][j]) {
            m[i][j] = q;
            s[i][j] = k;
          }
        }
      }
    }

    // We use the result from the algorithm to compute the matrix chain product via recursion
    return _chain_matmul_general(matrices, s, 0, n - 1);
  }
}

/*
Calculates the Kronecker product between two Tensors.
*/
Tensor& kron_out(const Tensor& self, const Tensor& other, Tensor& result) {
  int64_t maxdim = std::max(self.dim(), other.dim());
  int64_t pad_self = maxdim - self.dim();
  int64_t pad_other = maxdim - other.dim();
  c10::SmallVector<int64_t, 10> a_reshape(2 * maxdim);
  c10::SmallVector<int64_t, 10> b_reshape(2 * maxdim);
  c10::SmallVector<int64_t, 10> result_reshape(maxdim);
  for (int64_t i = 0; i < maxdim; i++) {
    a_reshape[2 * i] = (i >= pad_self ? self.sizes()[i - pad_self] : 1);
    a_reshape[2 * i + 1] = 1;
    b_reshape[2 * i] = 1;
    b_reshape[2 * i + 1] = (i >= pad_other ? other.sizes()[i - pad_other] : 1);
    result_reshape[i] = a_reshape[2 * i] * b_reshape[2 * i + 1];
  }
  auto self_view = at::_unsafe_view(self, a_reshape);
  auto other_view = at::_unsafe_view(other, b_reshape);
  if (!result.defined()) {
    result = at::_unsafe_view(at::mul(self_view, other_view), result_reshape);
  } else {
    c10::SmallVector<int64_t, 10> mul_shape(2 * maxdim);
    for (int64_t i = 0; i < maxdim; i++) {
      mul_shape[2 * i] = a_reshape[2 * i];
      mul_shape[2 * i + 1] = b_reshape[2 * i + 1];
    }
    resize_output(result, result_reshape);
    auto result_mul = at::_unsafe_view(result, mul_shape);
    at::mul_out(result_mul, self_view, other_view);
  }
  return result;
}

Tensor kron(const Tensor& self, const Tensor& other) {
  at::Tensor result;
  return at::kron_out(result, self, other);
}

} // namespace native
} // namespace at
