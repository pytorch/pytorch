#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/Gesv.h"

#include "TH.h"  // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgesv_(
    int* n, int* nrhs, double* a, int* lda,
    int *ipiv, double* b, int* ldb, int* info);
extern "C" void sgesv_(
    int* n, int* nrhs, float* a, int* lda,
    int* ipiv, float* b, int* ldb, int* info);
#endif

namespace at { namespace native {

template<class scalar_t>
void lapackGesv(
    int n, int nrhs, scalar_t* a, int lda, int* ipiv,
    scalar_t* b, int ldb, int* info) {
  AT_ERROR("gesv only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGesv<float>(
    int n, int nrhs, float* a, int lda, int* ipiv,
    float* b, int ldb, int* info) {
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGesv<double>(
    int n, int nrhs, double* a, int lda, int* ipiv,
    double* b, int ldb, int* info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}
#endif

static inline bool isTransposeContiguous(Tensor& self) {
 return self.dim() == 2 &&
        self.stride(0) == 1 &&
        self.stride(1) == self.size(0);
}

template <typename scalar_t>
static void applyGesv(Tensor& b, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#endif
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  auto ipiv = at::empty({n}, b.options().dtype(kInt));

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    lapackGesv<scalar_t>(n, nrhs, A_working_ptr, n, ipiv.data<int>(),
        b_working_ptr, n, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

std::tuple<Tensor&,Tensor&> _gesv_single_out_cpu(
    Tensor& sol, Tensor& lu,
    const Tensor& self, const Tensor& A) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#endif
  /* gesv takes two tensors (self, A) and returns (sol, lu).
   * The output Tensors (sol, lu) may be the same as input Tensors (self, A)
   *
   * Before passing pointers into Lapack, we need to ensure that:
   * (i)  self and A are represented in column major format
   * (ii) These pointers point to contiguous data for self and A.
   *
   * For 2D matrices, A.t() and self.t() represent their column major formats
   *
   * Case 1) The output tensor is of the correct shape, but it and its transpose
   *         are not contiguous. eg. torch.gesv(... , out=(n[::2], ...)):
   *         - clone input tensor into a buffer and use it
   *         - if output tensor is contiguous, it will be handled in case 3
   *
   * Note: In both cases below, we resize_ if required. This helps to:
   *       (i)  Make output tensors bigger and contiguous, if required, and
   *       (ii) Unsqueeze potential 1D `sol`, eg. torch.gesv(b, A, out=(b, A))
   *
   * Case 2) output_tensor.t() is contiguous (tc_sol / tc_lu is true):
   *         a) &input_tensor == &output_tensor:
   *            - it's fine to use output_tensor.data() as-is. Do nothing.
   *              (we need to transpose input_tensor for column-major anyway)
   *         b) &input_tensor != &output_tensor:
   *            - copy input_tensor.t() to output_tensor.t()
   *
   * Case 3) output_tensor.t() is not contiguous:
   *         - resize_ should make non-contig/incorrectly-sized tensors usable
   *         a) &input_tensor == &output_tensor:
   *            - clone and copy input_tensor.t() to output_tensor (same tensor)
   *         b) &input_tensor != &output_tensor:
   *            - copy input_tensor.t() to output_tensor
   */
  int64_t bx = self.size(0);
  int64_t by = (self.dim() == 1) ? 1 : self.size(1);
  int64_t ax = A.size(0);
  int64_t ay = A.size(1);
  int info = 0;
  bool tc_sol = isTransposeContiguous(sol);
  bool tc_lu = isTransposeContiguous(lu);
  bool sol_correct_shape = sol.dim() == 2 &&
                           sol.size(0) == bx && sol.size(1) == by;
  bool lu_correct_shape = lu.dim() == 2 && lu.size(0) == ax && lu.size(1) == ay;
  Tensor temp_sol;
  Tensor temp_lu;

  /* self is always viewable to {bx, by} since they are the dimensions
   * of self (or by == 1). Basically a shortcut to see 1D `self` as 2D */
  auto self_t = self.view({bx, by}).t_();

  if (!tc_sol && !sol.is_contiguous() && sol_correct_shape) {
    temp_sol = self_t.clone().t_();
  } else if (tc_sol) {
    sol.t().resize_({by, bx});
    if (&self != &sol) {
      sol.t().copy_(self_t);
    }
  } else {
    sol.resize_({by, bx});
    if (&self == &sol) {
      sol.copy_(self_t.clone()).t_();
    } else {
      sol.copy_(self_t).t_();
    }
  }

  if (!tc_lu && !lu.is_contiguous() && lu_correct_shape) {
    temp_lu = A.t().clone().t_();
  } else if (tc_lu) {
    lu.t().resize_({ay, ax});
    if (&A != &lu) {
      lu.t().copy_(A.t());
    }
  } else {
    lu.resize_({ay, ax});
    if (&A == &lu) {
      lu.copy_(A.t().clone()).t_();
    } else {
      lu.copy_(A.t()).t_();
    }
  }

  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    auto A_ptr = temp_lu.defined() ? temp_lu.data<scalar_t>()
                                   : lu.data<scalar_t>();
    auto b_ptr = temp_sol.defined() ? temp_sol.data<scalar_t>()
                                    : sol.data<scalar_t>();
    auto ipiv = at::empty({bx}, sol.options().dtype(kInt));
    lapackGesv<scalar_t>(bx, by, A_ptr, bx, ipiv.data<int>(), b_ptr, bx, &info);
  });

  checkErrors({info});

  if (temp_sol.defined()) {
    sol.copy_(temp_sol);
  }
  if (temp_lu.defined()) {
    lu.copy_(temp_lu);
  }

  return std::tuple<Tensor&, Tensor&>(sol, lu);
}

std::tuple<Tensor,Tensor> _gesv_helper_cpu(const Tensor& self, const Tensor& A) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto b_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    applyGesv<scalar_t>(b_working_copy, A_working_copy, infos);
  });
  checkErrors(infos);
  return std::tuple<Tensor,Tensor>(b_working_copy, A_working_copy);
}

std::tuple<Tensor,Tensor> _gesv_single(const Tensor& self, const Tensor& A) {
  auto A_ = self.type().tensor();
  auto b_ = self.type().tensor();
  return self.type()._gesv_single_out(b_, A_, self, A);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> gesv(const Tensor& self, const Tensor& A) {
  bool batched = !(self.dim() <= 2 && A.dim() <= 2);
  checkInputs(self, A, batched);

  if (!batched) {
    return at::_gesv_single(self, A);
  }

  // broadcast the batch dimensions of self and A.
  IntList self_batch_sizes(self.sizes().data(), self.ndimension() - 2);
  IntList A_batch_sizes(A.sizes().data(), A.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion =
      infer_size(self_batch_sizes, A_batch_sizes);

  std::vector<int64_t> self_expand_size({expand_batch_portion});
  self_expand_size.insert(self_expand_size.end(),
      { self.size(-2), self.size(-1) });

  std::vector<int64_t> A_expand_size({expand_batch_portion});
  A_expand_size.insert(A_expand_size.end(),
      { A.size(-2), A.size(-1) });

  Tensor self_broadcasted  = self.expand(self_expand_size);
  Tensor A_broadcasted = A.expand(A_expand_size);
  return self.type()._gesv_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> gesv_out(
    Tensor& sol, Tensor& lu, const Tensor& self, const Tensor& A) {
  if (self.dim() > 2 || A.dim() > 2) {
    AT_ERROR("torch.gesv() with the `out` keyword does not support batching. "
                  "b.dim() (%lld) and A.dim() (%lld) must both be 2.",
                  (long long)self.dim(), (long long)A.dim());
  }

  return self.type()._gesv_single_out(sol, lu, self, A);
}

}}  // namespace at::native
