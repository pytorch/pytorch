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

template <class scalar_t>
void lapackCholeskyInverse(char uplo, int n, scalar_t *a, int lda, int *info);

template <class scalar_t, class value_t=scalar_t>
void lapackEig(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *w, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, value_t *rwork, int *info);

template <class scalar_t>
void lapackGeqrf(int m, int n, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);

template <class scalar_t>
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

using geqrf_fn = void (*)(const Tensor& /*input*/, const Tensor& /*tau*/, int64_t /*m*/, int64_t /*n*/);
DECLARE_DISPATCH(geqrf_fn, geqrf_stub);

using orgqr_fn = Tensor& (*)(Tensor& /*result*/, const Tensor& /*tau*/, int64_t /*n_columns*/);
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
