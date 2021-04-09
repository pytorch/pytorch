#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/zmath.h>

#include <TH/TH.h> // for USE_LAPACK


namespace at { namespace native {

#ifdef USE_LAPACK
// Define per-batch functions to be used in the implementation of batched
// linear algebra operations

template<class scalar_t>
void lapackCholeskyInverse(char uplo, int n, scalar_t *a, int lda, int *info);

template<class scalar_t, class value_t=scalar_t>
void lapackEig(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *w, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, value_t *rwork, int *info);

template<class scalar_t>
void lapackOrgqr(int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);

template <class scalar_t, class value_t = scalar_t>
void lapackSyevd(char jobz, char uplo, int n, scalar_t* a, int lda, value_t* w, scalar_t* work, int lwork, value_t* rwork, int lrwork, int* iwork, int liwork, int* info);

template <class scalar_t>
void lapackTriangularSolve(char uplo, char trans, char diag, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb, int* info);

#endif

using cholesky_inverse_fn = Tensor& (*)(Tensor& /*result*/, Tensor& /*infos*/, bool /*upper*/);

DECLARE_DISPATCH(cholesky_inverse_fn, cholesky_inverse_stub);

using eig_fn = std::tuple<Tensor, Tensor> (*)(const Tensor&, bool&);

DECLARE_DISPATCH(eig_fn, eig_stub);

using linalg_eig_fn = void (*)(Tensor& /*eigenvalues*/, Tensor& /*eigenvectors*/, Tensor& /*infos*/, const Tensor& /*input*/, bool /*compute_eigenvectors*/);

DECLARE_DISPATCH(linalg_eig_fn, linalg_eig_stub);

/*
  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as produced by the geqrf function.

  Args:
  * `self` - Tensor with the directions of the elementary reflectors below the diagonal,
              it will be overwritten with the result
  * `tau` - Tensor containing the magnitudes of the elementary reflectors
  * `infos` - Tensor to store LAPACK's error codes
  * `n_columns` - The number of columns of Q to be computed

  For further details, please see the LAPACK documentation for ORGQR and UNGQR.
*/
template <typename scalar_t>
inline void apply_orgqr(Tensor& self, const Tensor& tau, Tensor& infos, int64_t n_columns) {
#ifndef USE_LAPACK
  TORCH_CHECK(false, "Calling torch.orgqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // Some LAPACK implementations might not work well with empty matrices:
  // workspace query might return lwork as 0, which is not allowed (requirement is lwork >= 1)
  // We don't need to do any calculations in this case, so let's return early
  if (self.numel() == 0) {
    infos.fill_(0);
    return;
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto self_matrix_stride = matrixStride(self);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(self);
  auto m = self.size(-2);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, m);

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n_columns);
  TORCH_INTERNAL_ASSERT(n_columns >= k);

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackOrgqr<scalar_t>(m, n_columns, k, self_data, lda, tau_data, &wkopt, lwork, &infos_data[0]);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());

  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    int* info_working_ptr = &infos_data[i];
    // now compute the actual Q
    lapackOrgqr<scalar_t>(m, n_columns, k, self_working_ptr, lda, tau_working_ptr, work.data_ptr<scalar_t>(), lwork, info_working_ptr);
    if (*info_working_ptr != 0) {
      return;
    }
  }
#endif
}

using orgqr_fn = Tensor& (*)(Tensor& /*result*/, const Tensor& /*tau*/, Tensor& /*infos*/, int64_t /*n_columns*/);
DECLARE_DISPATCH(orgqr_fn, orgqr_stub);

using linalg_eigh_fn = void (*)(
    Tensor& /*eigenvalues*/,
    Tensor& /*eigenvectors*/,
    Tensor& /*infos*/,
    bool /*upper*/,
    bool /*compute_eigenvectors*/);
DECLARE_DISPATCH(linalg_eigh_fn, linalg_eigh_stub);

using triangular_solve_fn = void (*)(
    Tensor& /*A*/,
    Tensor& /*b*/,
    Tensor& /*infos*/,
    bool /*upper*/,
    bool /*transpose*/,
    bool /*conjugate_transpose*/,
    bool /*unitriangular*/);
DECLARE_DISPATCH(triangular_solve_fn, triangular_solve_stub);

}} // namespace at::native
