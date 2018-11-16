#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/TensorUtils.h"
#include "ATen/Parallel.h"
#include <functional>
#include <numeric>
#include <vector>

namespace at {
namespace native {

// Helper function for det methods.
// For pivoted LU factorization A = P * L * U. Since we always have det(L) = 1,
// det(P) = \pm 1, this method returns a 3-tuple:
//   (det(P), diag(U), info),
// where info helps us identify singular matrices.
static inline std::tuple<double, Tensor, int> _lu_det_P_diag_U_info(const Tensor& self) {
  Tensor p, lu, info;
  std::tie(lu, p, info) = self.unsqueeze(0).btrifact_with_info();
  p.squeeze_(0);
  lu.squeeze_(0);
  int int_info = info.squeeze_().item<int32_t>();
  AT_CHECK(int_info >= 0, "LU factorization (getrf) failed with info = ", int_info);
  auto n = self.size(0);
  auto num_exchanges = (at::arange(1, n + 1, p.type()) != p).nonzero().size(0);
  if (num_exchanges % 2 == 1) {
    return std::make_tuple(-1., lu.diag(), int_info);
  } else {
    return std::make_tuple(1., lu.diag(), int_info);
  }
}

Tensor det(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "det(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    return at::zeros({}, self.type());
  } else {
    return diag_U.prod().mul_(det_P);
  }
}

Tensor logdet(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "logdet(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U, det;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    det = at::zeros({}, self.type());
  } else {
    det = diag_U.prod().mul_(det_P);
  }
  if (det.sign().item<double>() <= 0) {
    return det.log_();  // in order to get proper -inf (det=0) or nan (det<0)
  } else {
    return diag_U.abs().log().sum();
  }
}

std::tuple<Tensor, Tensor> slogdet(const Tensor& self) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) &&
           self.dim() == 2 && self.size(0) == self.size(1),
           "slogdet(", self.type(), "{", self.sizes(), "}): expected a 2D square tensor "
           "of floating types");
  double det_P;
  Tensor diag_U, det;
  int info;
  std::tie(det_P, diag_U, info) = _lu_det_P_diag_U_info(self);
  if (info > 0) {
    det = at::zeros({}, self.type());
  } else {
    det = diag_U.prod().mul_(det_P);
  }
  return std::make_tuple(det.sign(), diag_U.abs_().log_().sum());
}

Tensor pinverse(const Tensor& self, double rcond) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "pinverse(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");
  if (self.numel() == 0) {
    // Match NumPy
    return at::empty({self.size(1), self.size(0)}, self.options());
  }
  Tensor U, S, V;
  std::tie(U, S, V) = self.svd();
  Tensor max_val = S[0];
  Tensor S_pseudoinv = at::where(S > rcond * max_val, S.reciprocal(), at::zeros({}, self.options()));
  return V.mm(S_pseudoinv.diag().mm(U.t()));
}

static inline Tensor _matrix_rank_helper(const Tensor& self, bool symmetric) {
  Tensor S;
  if (!symmetric) {
    Tensor U, V;
    std::tie(U, S, V) = self.svd(/*some=*/true, /*compute_uv=*/false);
  } else {
    Tensor eigvecs;
    std::tie(S, eigvecs) = self.symeig(/*eigenvectors=*/false);
    S = S.abs();
  }
  return S;
}

Tensor matrix_rank(const Tensor& self, double tol, bool symmetric) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "matrix_rank(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");

  Tensor S = _matrix_rank_helper(self, symmetric);
  return (S > tol).sum();
}

Tensor matrix_rank(const Tensor& self, bool symmetric) {
  AT_CHECK(at::isFloatingType(self.type().scalarType()) && self.dim() == 2,
           "matrix_rank(", self.type(), "{", self.sizes(), "}): expected a 2D tensor "
           "of floating types");

  Tensor S = _matrix_rank_helper(self, symmetric);
  double tol = _get_epsilon(self.type().scalarType()) * std::max(self.size(0), self.size(1));
  return (S > S.max().mul_(tol)).sum();
}

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
 AT_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

Tensor ger(const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "ger");
  check_1d(vec2, "vec2", "ger");
  return at::_th_ger(self, vec2);
}

Tensor& ger_out(Tensor& result, const Tensor& self, const Tensor& vec2) {
  check_1d(self, "self", "ger");
  check_1d(vec2, "vec2", "ger");
  return at::_th_ger_out(result, self, vec2);
}

Tensor mm(const Tensor& self, const Tensor& mat2) {
  if (self.is_sparse()) {
    return mat2.type().addmm(at::zeros({}, mat2.type()), self, mat2, 0, 1);
  }
  return at::_th_mm(self, mat2);
}

Tensor& mm_out(Tensor& result, const Tensor& self, const Tensor& mat2) {
  if (self.is_sparse()) {
    return at::addmm_out(result, at::zeros({}, mat2.options()), self, mat2, 0, 1);
  }
  return at::_th_mm_out(result, self, mat2);
}

Tensor mv(const Tensor& self, const Tensor& vec) {
  check_1d(vec, "vec", "mv");
  return at::_th_mv(self, vec);
}

Tensor& mv_out(Tensor& result, const Tensor& self, const Tensor& vec) {
  check_1d(vec, "vec", "mv");
  return at::_th_mv_out(result, self, vec);
}

Tensor addmv(const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return at::_th_addmv(self, mat, vec, beta, alpha);
}

Tensor& addmv_(Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return at::_th_addmv_(self, mat, vec, beta, alpha);
}

Tensor& addmv_out(Tensor &result, const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  check_1d(vec, "vec", "addmv");
  return at::_th_addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor addr(const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_th_addr(self, vec1, vec2, beta, alpha);
}

Tensor& addr_(Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_th_addr_(self, vec1, vec2, beta, alpha);
}

Tensor& addr_out(Tensor &result, const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");
  return at::_th_addr_out(result, self, vec1, vec2, beta, alpha);
}

template <typename scalar_t, bool is_bmm>
inline void baddbmm_cpu_kernel(const Tensor& result, const Tensor& self, const Tensor& mat2, Scalar beta_, Scalar alpha_) {
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

static inline Tensor& bmm_out_or_baddbmm_(Tensor& self_or_result, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha, bool is_bmm_out) {
  // is_bmm_out: true for bmm_out, false for baddbmm_
  // self_or_result is "self" for baddbmm_ and "result" for bmm_out
  CheckedFrom c = (is_bmm_out ? "bmm" : "baddbmm");
  TensorArg self_arg(self_or_result, is_bmm_out ? "self" : "result", 0);
  TensorArg b1_arg(batch1, "batch1", 1);
  TensorArg b2_arg(batch2, "batch2", 2);
  checkBackend(c, {self_or_result, batch1, batch2}, Backend::CPU);
  checkDim(c, b1_arg, 3);
  checkDim(c, b2_arg, 3);

  int64_t bs = batch1.size(0);
  checkSize(c, b2_arg, 0, bs);
  int64_t contraction_size = batch1.size(2);
  int64_t res_rows = batch1.size(1);
  int64_t res_cols = batch2.size(2);
  checkSize(c, b2_arg, 1, contraction_size);

  if (is_bmm_out) {
    self_or_result.resize_({bs, res_rows, res_cols});
  } else {
    checkSize(c, self_arg, 0, bs);
    checkSize(c, self_arg, 1, res_rows);
    checkSize(c, self_arg, 2, res_cols);
  }

  // handle pathological cases that blas may not like
  if (self_or_result.numel() == 0) {
    return self_or_result;
  } else if (contraction_size == 0) {
    return self_or_result.zero_();
  }

  auto batch_items_contiguous_or_transposed = [&](const Tensor& t) {
    return (t.stride(2) == 1 && t.stride(1) == t.size(2))
            || (t.stride(1) == 1 && t.stride(2) == t.size(1));
  };

  if (contraction_size * res_rows * res_cols < 400) {
    if (is_bmm_out) {
      AT_DISPATCH_ALL_TYPES(batch1.type(), "bmm", [&] {
          baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
        });
    } else {
      AT_DISPATCH_ALL_TYPES(batch1.type(), "baddbmm", [&] {
          baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
        });
    }
  } else if (at::hasMKL() && at::native::is_floating_point(self_or_result)
	     && batch_items_contiguous_or_transposed(batch1)
	     && batch_items_contiguous_or_transposed(batch2)
	     && self_or_result.is_contiguous()) {
    at::native::_baddbmm_mkl_(self_or_result, batch1, batch2, beta, alpha);
  } else { // split along batch dimension
    if (is_bmm_out) {
      for (int64_t b = 0; b < bs; b++) {
        auto r = self_or_result.select(0, b);
        at::native::mm_out(r, batch1.select(0, b), batch2.select(0, b));
      }
    } else {
      for (int64_t b = 0; b < bs; b++) {
        self_or_result.select(0, b).addmm_(batch1.select(0, b), batch2.select(0, b), beta, alpha);
      }
    }
  }
  return self_or_result;
}


Tensor baddbmm_cpu(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({0}, self.options());
  return at::native::baddbmm_out_cpu(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_cpu(Tensor &result, const Tensor& self_, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  Tensor self;
  std::tie(self) = expand_size(self_, {batch1.size(0), batch1.size(1), batch2.size(2)}, "baddbmm");
  result.resize_(self.sizes());
  result.copy_(self);
  return at::native::baddbmm__cpu(result, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cpu(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return bmm_out_or_baddbmm_(self, batch1, batch2, beta, alpha, false);
}

Tensor bmm_cpu(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({0}, self.options());
  return at::native::bmm_out_cpu(result, self, mat2);
}

Tensor& bmm_out_cpu(Tensor &result, const Tensor& batch1, const Tensor& batch2) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  return bmm_out_or_baddbmm_(result, batch1, batch2, beta, alpha, true);
}

Tensor dot(const Tensor& self, const Tensor& tensor) {
  check_1d(self, "self", "dot");
  check_1d(tensor, "tensor", "dot");
  return at::_th_dot(self, tensor);
}

Tensor& dot_out(Tensor& result, const Tensor& self, const Tensor& tensor) {
  result.resize_({});
  // dispatching through type ensures we don't allow mismatched types.
  return self.type().fill_(result, self.dot(tensor));
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
  auto dim_tensor1 = tensor1.dim();
  auto dim_tensor2 = tensor2.dim();
  auto has_out = out_opt.has_value();
  Tensor out = out_opt.value_or(Tensor());

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::native::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::native::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::native::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::native::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
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
  } else if ((dim_tensor1 >= 1 && dim_tensor2 >= 1) && (dim_tensor1 >= 3 || dim_tensor2 >= 3)) {
    // We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
    // we track m1 vs m2 separately even though they must match for nicer error messages
    int64_t n = dim_tensor1 > 1 ? tensor1.size(-2) : 1;
    int64_t m1 = tensor1.size(-1);
    IntList batch_tensor1(tensor1.sizes().data(), std::max<int64_t>(dim_tensor1 - 2, 0));
    int64_t m2 = dim_tensor2 > 1 ? tensor2.size(-2) : 1;
    int64_t p = tensor2.size(-1);
    IntList batch_tensor2(tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0));

    // expand the batch portion (i.e. cut off matrix dimensions and expand rest)
    std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);

    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {n, m1});

    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {m2, p});

    int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(),
                                               1, std::multiplies<int64_t>());

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
  return at::native::matmul(c10::nullopt, tensor1, tensor2);
}

Tensor& matmul_out(Tensor &result, const Tensor & tensor1, const Tensor & tensor2) {
  at::native::matmul(c10::optional<Tensor>(result), tensor1, tensor2);
  return result;
}

Tensor matrix_power(const Tensor& a, int64_t n) {
  AT_CHECK(a.dim() >= 2 && at::isFloatingType(a.type().scalarType()),
           "matrix_power(", a.type(), "{", a.sizes(), "}): expected a tensor "
           "of floating types with dim at least 2");
  if (n == 0) {
    return a.clone().copy_(at::eye(a.size(-2), a.options()).expand_as(a));
  } else if (n < 0) {
    Tensor a_ = at::inverse(a);
    n *= -1;
    return at::native::matrix_power(a_, n);
  } else if (n == 1) {
    return a.clone();
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
    z = (!z.defined()) ? a.clone() : at::native::matmul(z, z);
    r = n % 2;
    n = n / 2;
    if (r == 1) {
      result = (!result.defined()) ? z.clone() : at::native::matmul(result, z);
    }
  }
  return result;
}

Tensor frobenius_norm(const Tensor& self) {
  return at::norm(self);
}

Tensor frobenius_norm(const Tensor& self, IntList dim, bool keepdim) {
  AT_CHECK(
      dim.size() <= 2,
      "Expected at most 2 dimensions, but got ",
      dim.size(),
      " dimensions instead.");
  if (dim.size() == 1) {
    return at::norm(self, 2, dim[0], keepdim);
  }
  return at::sqrt(at::sum(self * self, dim, keepdim));
}

Tensor &frobenius_norm_out(
    Tensor& result,
    const Tensor& self,
    IntList dim,
    bool keepdim) {
  AT_CHECK(
      dim.size() <= 2,
      "Expected at most 2 dimensions, but got ",
      dim.size(),
      " dimensions instead.");
  if (dim.size() == 1) {
    return at::norm_out(result, self, 2, dim[0], keepdim);
  }
  return at::sqrt_out(result, at::sum(self * self, dim, keepdim));
}

Tensor nuclear_norm(const Tensor& self, bool keepdim) {
  AT_CHECK(
      self.dim() == 2,
      "Expected a tensor with 2 dimensions, but got a ",
      self.dim(),
      " dimensions tensor instead.");
  return at::sum(std::get<1>(at::svd(self)), 0, keepdim);
}

Tensor &nuclear_norm_out(Tensor& result, const Tensor& self, bool keepdim) {
  AT_CHECK(
      self.dim() == 2,
      "Expected a tensor with 2 dimensions, but got a ",
      self.dim(),
      " dimensions tensor instead.");
  return at::sum_out(result, std::get<1>(at::svd(self)), 0, keepdim);
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
    for (int64_t i = 0; i < n; i++) {
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

} // namespace native
} // namespace at
