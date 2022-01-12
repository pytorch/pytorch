#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TransposeType.h>


namespace at { namespace native {

enum class LapackLstsqDriverType : int64_t { Gels, Gelsd, Gelsy, Gelss};

#if AT_BUILD_WITH_LAPACK()
// Define per-batch functions to be used in the implementation of batched
// linear algebra operations

template <class scalar_t>
void lapackCholesky(char uplo, int n, scalar_t *a, int lda, int *info);

template <class scalar_t>
void lapackCholeskyInverse(char uplo, int n, scalar_t *a, int lda, int *info);

template <class scalar_t, class value_t=scalar_t>
void lapackEig(char jobvl, char jobvr, int n, scalar_t *a, int lda, scalar_t *w, scalar_t* vl, int ldvl, scalar_t *vr, int ldvr, scalar_t *work, int lwork, value_t *rwork, int *info);

template <class scalar_t>
void lapackGeqrf(int m, int n, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);

template <class scalar_t>
void lapackOrgqr(int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *work, int lwork, int *info);

template <class scalar_t>
void lapackOrmqr(char side, char trans, int m, int n, int k, scalar_t *a, int lda, scalar_t *tau, scalar_t *c, int ldc, scalar_t *work, int lwork, int *info);

template <class scalar_t, class value_t = scalar_t>
void lapackSyevd(char jobz, char uplo, int n, scalar_t* a, int lda, value_t* w, scalar_t* work, int lwork, value_t* rwork, int lrwork, int* iwork, int liwork, int* info);

template <class scalar_t>
void lapackGels(char trans, int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    scalar_t *work, int lwork, int *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelsd(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    value_t *s, value_t rcond, int *rank,
    scalar_t* work, int lwork,
    value_t *rwork, int* iwork, int *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelsy(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    int *jpvt, value_t rcond, int *rank,
    scalar_t *work, int lwork, value_t* rwork, int *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelss(int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    value_t *s, value_t rcond, int *rank,
    scalar_t *work, int lwork,
    value_t *rwork, int *info);

template <LapackLstsqDriverType, class scalar_t, class value_t = scalar_t>
struct lapackLstsq_impl;

template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gels, scalar_t, value_t> {
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    lapackGels<scalar_t>(
        trans, m, n, nrhs,
        a, lda, b, ldb,
        work, lwork, info);
  }
};

template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelsy, scalar_t, value_t> {
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    lapackGelsy<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        jpvt, rcond, rank,
        work, lwork, rwork, info);
  }
};

template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelsd, scalar_t, value_t> {
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    lapackGelsd<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        s, rcond, rank,
        work, lwork,
        rwork, iwork, info);
  }
};

template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gelss, scalar_t, value_t> {
  static void call(
      char trans, int m, int n, int nrhs,
      scalar_t *a, int lda, scalar_t *b, int ldb,
      scalar_t *work, int lwork, int *info, // Gels flavor
      int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      int *iwork // Gelsd flavor
      ) {
    lapackGelss<scalar_t, value_t>(
        m, n, nrhs,
        a, lda, b, ldb,
        s, rcond, rank,
        work, lwork,
        rwork, info);
  }
};

template <LapackLstsqDriverType driver_type, class scalar_t, class value_t = scalar_t>
void lapackLstsq(
    char trans, int m, int n, int nrhs,
    scalar_t *a, int lda, scalar_t *b, int ldb,
    scalar_t *work, int lwork, int *info, // Gels flavor
    int *jpvt, value_t rcond, int *rank, value_t* rwork, // Gelsy flavor
    value_t *s, // Gelss flavor
    int *iwork // Gelsd flavor
    ) {
  lapackLstsq_impl<driver_type, scalar_t, value_t>::call(
      trans, m, n, nrhs,
      a, lda, b, ldb,
      work, lwork, info,
      jpvt, rcond, rank, rwork,
      s,
      iwork);
}

template <class scalar_t>
void lapackLuSolve(char trans, int n, int nrhs, scalar_t *a, int lda, int *ipiv, scalar_t *b, int ldb, int *info);

template <class scalar_t>
void lapackLu(int m, int n, scalar_t *a, int lda, int *ipiv, int *info);

#endif

#if AT_BUILD_WITH_BLAS()
template <class scalar_t>
void blasTriangularSolve(char side, char uplo, char trans, char diag, int n, int nrhs, scalar_t* a, int lda, scalar_t* b, int ldb);
#endif

using cholesky_fn = void (*)(const Tensor& /*input*/, const Tensor& /*info*/, bool /*upper*/);
DECLARE_DISPATCH(cholesky_fn, cholesky_stub);

using cholesky_inverse_fn = Tensor& (*)(Tensor& /*result*/, Tensor& /*infos*/, bool /*upper*/);

DECLARE_DISPATCH(cholesky_inverse_fn, cholesky_inverse_stub);

using eig_fn = std::tuple<Tensor, Tensor> (*)(const Tensor&, bool&);

DECLARE_DISPATCH(eig_fn, eig_stub);

using linalg_eig_fn = void (*)(Tensor& /*eigenvalues*/, Tensor& /*eigenvectors*/, Tensor& /*infos*/, const Tensor& /*input*/, bool /*compute_eigenvectors*/);

DECLARE_DISPATCH(linalg_eig_fn, linalg_eig_stub);

using geqrf_fn = void (*)(const Tensor& /*input*/, const Tensor& /*tau*/);
DECLARE_DISPATCH(geqrf_fn, geqrf_stub);

using orgqr_fn = Tensor& (*)(Tensor& /*result*/, const Tensor& /*tau*/);
DECLARE_DISPATCH(orgqr_fn, orgqr_stub);

using ormqr_fn = void (*)(const Tensor& /*input*/, const Tensor& /*tau*/, const Tensor& /*other*/, bool /*left*/, bool /*transpose*/);
DECLARE_DISPATCH(ormqr_fn, ormqr_stub);

using linalg_eigh_fn = void (*)(
    const Tensor& /*eigenvalues*/,
    const Tensor& /*eigenvectors*/,
    const Tensor& /*infos*/,
    bool /*upper*/,
    bool /*compute_eigenvectors*/);
DECLARE_DISPATCH(linalg_eigh_fn, linalg_eigh_stub);

using lstsq_fn = void (*)(
    const Tensor& /*a*/,
    Tensor& /*b*/,
    Tensor& /*rank*/,
    Tensor& /*singular_values*/,
    Tensor& /*infos*/,
    double /*rcond*/,
    std::string /*driver_name*/);
DECLARE_DISPATCH(lstsq_fn, lstsq_stub);

using triangular_solve_fn = void (*)(
    const Tensor& /*A*/,
    const Tensor& /*B*/,
    bool /*left*/,
    bool /*upper*/,
    TransposeType /*transpose*/,
    bool /*unitriangular*/);
DECLARE_DISPATCH(triangular_solve_fn, triangular_solve_stub);

using lu_factor_fn = void (*)(
    const Tensor& /*input*/,
    const Tensor& /*pivots*/,
    const Tensor& /*infos*/,
    bool /*compute_pivots*/);
DECLARE_DISPATCH(lu_factor_fn, lu_factor_stub);

using lu_solve_fn = void (*)(
    const Tensor& /*b*/,
    const Tensor& /*lu*/,
    const Tensor& /*pivots*/);
DECLARE_DISPATCH(lu_solve_fn, lu_solve_stub);

using lu_solve_trans_fn = void (*)(
    const Tensor& /*b*/,
    const Tensor& /*lu*/,
    const Tensor& /*pivots*/,
    TransposeType /*trans*/);
DECLARE_DISPATCH(lu_solve_trans_fn, lu_solve_trans_stub);


}} // namespace at::native
