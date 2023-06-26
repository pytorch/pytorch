#pragma once

#include <c10/util/Optional.h>
#include <ATen/Config.h>
#include <ATen/native/DispatchStub.h>

#if AT_MKL_ENABLED()
#if MKL_ILP64
#define INT_T int64_t
#else
#define INT_T int32_t
#endif
#else
#ifdef OPENBLAS_USE64BITINT
#define INT_T int64_t
#else
#define INT_T int
#endif
#endif

// Forward declare TI
namespace at {
class Tensor;
struct TensorIterator;

namespace native {
enum class TransposeType;
}

}

namespace at::native {

enum class LapackLstsqDriverType : int64_t { Gels, Gelsd, Gelsy, Gelss};

#if AT_BUILD_WITH_LAPACK()
// Define per-batch functions to be used in the implementation of batched
// linear algebra operations

template <class scalar_t>
void lapackCholesky(char uplo, INT_T n, scalar_t *a, INT_T lda, INT_T *info);

template <class scalar_t>
void lapackCholeskyInverse(char uplo, INT_T n, scalar_t *a, INT_T lda, INT_T *info);

template <class scalar_t, class value_t=scalar_t>
void lapackEig(char jobvl, char jobvr, INT_T n, scalar_t *a, INT_T lda, scalar_t *w, scalar_t* vl, INT_T ldvl, scalar_t *vr, INT_T ldvr, scalar_t *work, INT_T lwork, value_t *rwork, INT_T *info);

template <class scalar_t>
void lapackGeqrf(INT_T m, INT_T n, scalar_t *a, INT_T lda, scalar_t *tau, scalar_t *work, INT_T lwork, INT_T *info);

template <class scalar_t>
void lapackOrgqr(INT_T m, INT_T n, INT_T k, scalar_t *a, INT_T lda, scalar_t *tau, scalar_t *work, INT_T lwork, INT_T *info);

template <class scalar_t>
void lapackOrmqr(char side, char trans, INT_T m, INT_T n, INT_T k, scalar_t *a, INT_T lda, scalar_t *tau, scalar_t *c, INT_T ldc, scalar_t *work, INT_T lwork, INT_T *info);

template <class scalar_t, class value_t = scalar_t>
void lapackSyevd(char jobz, char uplo, INT_T n, scalar_t* a, INT_T lda, value_t* w, scalar_t* work, INT_T lwork, value_t* rwork, INT_T lrwork, INT_T* iwork, INT_T liwork, INT_T* info);

template <class scalar_t>
void lapackGels(char trans, INT_T m, INT_T n, INT_T nrhs,
    scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
    scalar_t *work, INT_T lwork, INT_T *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelsd(INT_T m, INT_T n, INT_T nrhs,
    scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
    value_t *s, value_t rcond, INT_T *rank,
    scalar_t* work, INT_T lwork,
    value_t *rwork, INT_T* iwork, INT_T *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelsy(INT_T m, INT_T n, INT_T nrhs,
    scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
    INT_T *jpvt, value_t rcond, INT_T *rank,
    scalar_t *work, INT_T lwork, value_t* rwork, INT_T *info);

template <class scalar_t, class value_t = scalar_t>
void lapackGelss(INT_T m, INT_T n, INT_T nrhs,
    scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
    value_t *s, value_t rcond, INT_T *rank,
    scalar_t *work, INT_T lwork,
    value_t *rwork, INT_T *info);

template <LapackLstsqDriverType, class scalar_t, class value_t = scalar_t>
struct lapackLstsq_impl;

template <class scalar_t, class value_t>
struct lapackLstsq_impl<LapackLstsqDriverType::Gels, scalar_t, value_t> {
  static void call(
      char trans, INT_T m, INT_T n, INT_T nrhs,
      scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
      scalar_t *work, INT_T lwork, INT_T *info, // Gels flavor
      INT_T *jpvt, value_t rcond, INT_T *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      INT_T *iwork // Gelsd flavor
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
      char trans, INT_T m, INT_T n, INT_T nrhs,
      scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
      scalar_t *work, INT_T lwork, INT_T *info, // Gels flavor
      INT_T *jpvt, value_t rcond, INT_T *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      INT_T *iwork // Gelsd flavor
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
      char trans, INT_T m, INT_T n, INT_T nrhs,
      scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
      scalar_t *work, INT_T lwork, INT_T *info, // Gels flavor
      INT_T *jpvt, value_t rcond, INT_T *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      INT_T *iwork // Gelsd flavor
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
      char trans, INT_T m, INT_T n, INT_T nrhs,
      scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
      scalar_t *work, INT_T lwork, INT_T *info, // Gels flavor
      INT_T *jpvt, value_t rcond, INT_T *rank, value_t* rwork, // Gelsy flavor
      value_t *s, // Gelss flavor
      INT_T *iwork // Gelsd flavor
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
    char trans, INT_T m, INT_T n, INT_T nrhs,
    scalar_t *a, INT_T lda, scalar_t *b, INT_T ldb,
    scalar_t *work, INT_T lwork, INT_T *info, // Gels flavor
    INT_T *jpvt, value_t rcond, INT_T *rank, value_t* rwork, // Gelsy flavor
    value_t *s, // Gelss flavor
    INT_T *iwork // Gelsd flavor
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
void lapackLuSolve(char trans, INT_T n, INT_T nrhs, scalar_t *a, INT_T lda, INT_T *ipiv, scalar_t *b, INT_T ldb, INT_T *info);

template <class scalar_t>
void lapackLu(INT_T m, INT_T n, scalar_t *a, INT_T lda, INT_T *ipiv, INT_T *info);

template <class scalar_t>
void lapackLdlHermitian(
    char uplo,
    INT_T n,
    scalar_t* a,
    INT_T lda,
    INT_T* ipiv,
    scalar_t* work,
    INT_T lwork,
    INT_T* info);

template <class scalar_t>
void lapackLdlSymmetric(
    char uplo,
    INT_T n,
    scalar_t* a,
    INT_T lda,
    INT_T* ipiv,
    scalar_t* work,
    INT_T lwork,
    INT_T* info);

template <class scalar_t>
void lapackLdlSolveHermitian(
    char uplo,
    INT_T n,
    INT_T nrhs,
    scalar_t* a,
    INT_T lda,
    INT_T* ipiv,
    scalar_t* b,
    INT_T ldb,
    INT_T* info);

template <class scalar_t>
void lapackLdlSolveSymmetric(
    char uplo,
    INT_T n,
    INT_T nrhs,
    scalar_t* a,
    INT_T lda,
    INT_T* ipiv,
    scalar_t* b,
    INT_T ldb,
    INT_T* info);

template<class scalar_t, class value_t=scalar_t>
void lapackSvd(char jobz, INT_T m, INT_T n, scalar_t *a, INT_T lda, value_t *s, scalar_t *u, INT_T ldu, scalar_t *vt, INT_T ldvt, scalar_t *work, INT_T lwork, value_t *rwork, INT_T *iwork, INT_T *info);
#endif

#if AT_BUILD_WITH_BLAS()
template <class scalar_t>
void blasTriangularSolve(char side, char uplo, char trans, char diag, INT_T n, INT_T nrhs, scalar_t* a, INT_T lda, scalar_t* b, INT_T ldb);
#endif

using cholesky_fn = void (*)(const Tensor& /*input*/, const Tensor& /*info*/, bool /*upper*/);
DECLARE_DISPATCH(cholesky_fn, cholesky_stub);

using cholesky_inverse_fn = Tensor& (*)(Tensor& /*result*/, Tensor& /*infos*/, bool /*upper*/);

DECLARE_DISPATCH(cholesky_inverse_fn, cholesky_inverse_stub);

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

using unpack_pivots_fn = void(*)(
  TensorIterator& iter,
  const int64_t dim_size,
  const int64_t max_pivot);
DECLARE_DISPATCH(unpack_pivots_fn, unpack_pivots_stub);

using lu_solve_fn = void (*)(
    const Tensor& /*LU*/,
    const Tensor& /*pivots*/,
    const Tensor& /*B*/,
    TransposeType /*trans*/);
DECLARE_DISPATCH(lu_solve_fn, lu_solve_stub);

using ldl_factor_fn = void (*)(
    const Tensor& /*LD*/,
    const Tensor& /*pivots*/,
    const Tensor& /*info*/,
    bool /*upper*/,
    bool /*hermitian*/);
DECLARE_DISPATCH(ldl_factor_fn, ldl_factor_stub);

using svd_fn = void (*)(
    const Tensor& /*A*/,
    const bool /*full_matrices*/,
    const bool /*compute_uv*/,
    const c10::optional<c10::string_view>& /*driver*/,
    const Tensor& /*U*/,
    const Tensor& /*S*/,
    const Tensor& /*Vh*/,
    const Tensor& /*info*/);
DECLARE_DISPATCH(svd_fn, svd_stub);

using ldl_solve_fn = void (*)(
    const Tensor& /*LD*/,
    const Tensor& /*pivots*/,
    const Tensor& /*result*/,
    bool /*upper*/,
    bool /*hermitian*/);
DECLARE_DISPATCH(ldl_solve_fn, ldl_solve_stub);
} // namespace at::native
