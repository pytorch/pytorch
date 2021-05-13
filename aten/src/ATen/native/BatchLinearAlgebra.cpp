#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/Parallel.h>

#include <c10/util/irange.h>

#include <TH/TH.h>  // for USE_LAPACK

#include <vector>

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#ifdef USE_LAPACK

// gesv
extern "C" void zgesv_(int *n, int *nrhs, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *b, int *ldb, int *info);
extern "C" void cgesv_(int *n, int *nrhs, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *b, int *ldb, int *info);
extern "C" void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// getrf
extern "C" void zgetrf_(int *m, int *n, std::complex<double> *a, int *lda, int *ipiv, int *info);
extern "C" void cgetrf_(int *m, int *n, std::complex<float> *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);

// getri
extern "C" void zgetri_(int *n, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *work, int *lwork, int *info);
extern "C" void cgetri_(int *n, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *work, int *lwork, int *info);
extern "C" void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern "C" void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);

// potrs
extern "C" void zpotrs_(char *uplo, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);
extern "C" void cpotrs_(char *uplo, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info);
extern "C" void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);

// potrf
extern "C" void zpotrf_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotrf_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

// potri
extern "C" void zpotri_(char *uplo, int *n, std::complex<double> *a, int *lda, int *info);
extern "C" void cpotri_(char *uplo, int *n, std::complex<float> *a, int *lda, int *info);
extern "C" void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotri_(char *uplo, int *n, float *a, int *lda, int *info);

// trtrs
extern "C" void ztrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);
extern "C" void ctrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb, int *info);
extern "C" void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void strtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);

// geqrf
extern "C" void zgeqrf_(int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cgeqrf_(int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// orgqr
extern "C" void zungqr_(int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *work, int *lwork, int *info);
extern "C" void cungqr_(int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *work, int *lwork, int *info);
extern "C" void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern "C" void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

// ormqr
extern "C" void zunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<double> *a, int *lda, std::complex<double> *tau, std::complex<double> *c, int *ldc, std::complex<double> *work, int *lwork, int *info);
extern "C" void cunmqr_(char *side, char *trans, int *m, int *n, int *k, std::complex<float> *a, int *lda, std::complex<float> *tau, std::complex<float> *c, int *ldc, std::complex<float> *work, int *lwork, int *info);
extern "C" void dormqr_(char *side, char *trans, int *m, int *n, int *k, double *a, int *lda, double *tau, double *c, int *ldc, double *work, int *lwork, int *info);
extern "C" void sormqr_(char *side, char *trans, int *m, int *n, int *k, float *a, int *lda, float *tau, float *c, int *ldc, float *work, int *lwork, int *info);

// syev
extern "C" void zheev_(char *jobz, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work, int *lwork, double *rwork, int *info);
extern "C" void cheev_(char *jobz, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work, int *lwork, float *rwork, int *info);
extern "C" void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *info);
extern "C" void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *info);

// syevd
extern "C" void zheevd_(char *jobz, char *uplo, int *n, std::complex<double> *a, int *lda, double *w, std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void cheevd_(char *jobz, char *uplo, int *n, std::complex<float> *a, int *lda, float *w, std::complex<float> *work, int *lwork, float *rwork, int *lrwork, int *iwork, int *liwork, int *info);
extern "C" void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w, double *work, int *lwork, int *iwork, int *liwork, int *info);
extern "C" void ssyevd_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w, float *work, int *lwork, int *iwork, int *liwork, int *info);

// geev
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr, double *wi, double* vl, int *ldvl, double *vr, int *ldvr, double *work, int *lwork, int *info);
extern "C" void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr, float *wi, float* vl, int *ldvl, float *vr, int *ldvr, float *work, int *lwork, int *info);
extern "C" void cgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<float> *a, int *lda,
             std::complex<float> *w,
             std::complex<float> *vl, int *ldvl,
             std::complex<float> *vr, int *ldvr,
             std::complex<float> *work, int *lwork,
             float *rwork,
             int *info);
extern "C" void zgeev_(char *jobvl, char *jobvr, int *n,
             std::complex<double> *a, int *lda,
             std::complex<double> *w,
             std::complex<double> *vl, int *ldvl,
             std::complex<double> *vr, int *ldvr,
             std::complex<double> *work, int *lwork,
             double *rwork,
             int *info);

// gesdd
extern "C" void zgesdd_(char *jobz, int *m, int *n, std::complex<double> *a, int *lda,
                        double *s, std::complex<double> *u, int *ldu, std::complex<double> *vt, int *ldvt, std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgesdd_(char *jobz, int *m, int *n, std::complex<float> *a, int *lda,
                        float *s, std::complex<float> *u, int *ldu, std::complex<float> *vt, int *ldvt, std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
                        double *s, double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info);
extern "C" void sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
                        float *s, float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info);

// getrs
extern "C" void zgetrs_(char *trans, int *n, int *nrhs, std::complex<double> *a, int *lda, int *ipiv, std::complex<double> *b, int *ldb, int *info);
extern "C" void cgetrs_(char *trans, int *n, int *nrhs, std::complex<float> *a, int *lda, int *ipiv, std::complex<float> *b, int *ldb, int *info);
extern "C" void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// gels
extern "C" void zgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    std::complex<double> *work, int *lwork, int *info);
extern "C" void cgels_(char *trans, int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    std::complex<float> *work, int *lwork, int *info);
extern "C" void dgels_(char *trans, int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *work, int *lwork, int *info);
extern "C" void sgels_(char *trans, int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *work, int *lwork, int *info);

// gelsd
extern "C" void zgelsd_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork, double *rwork, int *iwork, int *info);
extern "C" void cgelsd_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork, float *rwork, int *iwork, int *info);
extern "C" void dgelsd_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *iwork, int *info);
extern "C" void sgelsd_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *iwork, int *info);

// gelsy
extern "C" void zgelsy_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelsy_(int *m, int *n, int *nrhs,
    std::complex<float> * a, int *lda, std::complex<float> *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelsy_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    int *jpvt, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelsy_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    int *jpvt, float *rcond, int *rank,
    float *work, int *lwork, int *info);

// gelss
extern "C" void zgelss_(int *m, int *n, int *nrhs,
    std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb,
    double *s, double *rcond, int *rank,
    std::complex<double> *work, int *lwork,
    double *rwork, int *info);
extern "C" void cgelss_(int *m, int *n, int *nrhs,
    std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb,
    float *s, float *rcond, int *rank,
    std::complex<float> *work, int *lwork,
    float *rwork, int *info);
extern "C" void dgelss_(int *m, int *n, int *nrhs,
    double *a, int *lda, double *b, int *ldb,
    double *s, double *rcond, int *rank,
    double *work, int *lwork, int *info);
extern "C" void sgelss_(int *m, int *n, int *nrhs,
    float *a, int *lda, float *b, int *ldb,
    float *s, float *rcond, int *rank,
    float *work, int *lwork, int *info);
#endif

namespace at {
namespace native {

#ifdef USE_LAPACK
// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations
template<class scalar_t>
void lapackSolve(int n, int nrhs, scalar_t *a, int lda, int *ipiv, scalar_t *b, int ldb, int *info);

template<class scalar_t>
void lapackGetri(int n, scalar_t *a, int lda, int *ipiv, scalar_t *work, int lwork, int *info);

template<class scalar_t>
void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info);

template<class scalar_t, class value_t=scalar_t>
void lapackSymeig(char jobz, char uplo, int n, scalar_t *a, int lda, value_t *w, scalar_t *work, int lwork, value_t *rwork, int *info);

template<class scalar_t, class value_t=scalar_t>
void lapackSvd(char jobz, int m, int n, scalar_t *a, int lda,
               value_t *s, scalar_t *u, int ldu, scalar_t *vt, int ldvt, scalar_t *work, int lwork, value_t *rwork, int *iwork, int *info);

template<> void lapackSolve<c10::complex<double>>(int n, int nrhs, c10::complex<double> *a, int lda, int *ipiv, c10::complex<double> *b, int ldb, int *info) {
  zgesv_(&n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackSolve<c10::complex<float>>(int n, int nrhs, c10::complex<float> *a, int lda, int *ipiv, c10::complex<float> *b, int ldb, int *info) {
  cgesv_(&n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackSolve<double>(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int *info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackSolve<float>(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int *info) {
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGetri<c10::complex<double>>(int n, c10::complex<double> *a, int lda, int *ipiv, c10::complex<double> *work, int lwork, int *info) {
  zgetri_(&n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGetri<c10::complex<float>>(int n, c10::complex<float> *a, int lda, int *ipiv, c10::complex<float> *work, int lwork, int *info) {
  cgetri_(&n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGetri<double>(int n, double *a, int lda, int *ipiv, double *work, int lwork, int *info) {
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetri<float>(int n, float *a, int lda, int *ipiv, float *work, int lwork, int *info) {
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackLu<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, int *ipiv, int *info) {
  zgetrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, int *ipiv, int *info) {
  cgetrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, info);
}

template<> void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackCholeskySolve<c10::complex<double>>(char uplo, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, int *info) {
  zpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<c10::complex<float>>(char uplo, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, int *info) {
  cpotrs_(&uplo, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackCholeskySolve<double>(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholeskySolve<float>(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholesky<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotrf_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholesky<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotrf_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  spotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<double>>(char uplo, int n, c10::complex<double> *a, int lda, int *info) {
  zpotri_(&uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<c10::complex<float>>(char uplo, int n, c10::complex<float> *a, int lda, int *info) {
  cpotri_(&uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, info);
}

template<> void lapackCholeskyInverse<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholeskyInverse<float>(char uplo, int n, float *a, int lda, int *info) {
  spotri_(&uplo, &n, a, &lda, info);
}

template<> void lapackTriangularSolve<c10::complex<double>>(char uplo, char trans, char diag, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb, int *info) {
  ztrtrs_(&uplo, &trans, &diag, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackTriangularSolve<c10::complex<float>>(char uplo, char trans, char diag, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb, int *info) {
  ctrtrs_(&uplo, &trans, &diag, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackTriangularSolve<double>(char uplo, char trans, char diag, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackTriangularSolve<float>(char uplo, char trans, char diag, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackGeqrf<c10::complex<double>>(int m, int n, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zgeqrf_(&m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGeqrf<c10::complex<float>>(int m, int n, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cgeqrf_(&m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGeqrf<double>(int m, int n, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackGeqrf<float>(int m, int n, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<c10::complex<double>>(int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *work, int lwork, int *info) {
  zungqr_(&m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrgqr<c10::complex<float>>(int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *work, int lwork, int *info) {
  cungqr_(&m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrgqr<double>(int m, int n, int k, double *a, int lda, double *tau, double *work, int lwork, int *info) {
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrgqr<float>(int m, int n, int k, float *a, int lda, float *tau, float *work, int lwork, int *info) {
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template<> void lapackOrmqr<c10::complex<double>>(char side, char trans, int m, int n, int k, c10::complex<double> *a, int lda, c10::complex<double> *tau, c10::complex<double> *c, int ldc, c10::complex<double> *work, int lwork, int *info) {
  zunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(tau), reinterpret_cast<std::complex<double>*>(c), &ldc, reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackOrmqr<c10::complex<float>>(char side, char trans, int m, int n, int k, c10::complex<float> *a, int lda, c10::complex<float> *tau, c10::complex<float> *c, int ldc, c10::complex<float> *work, int lwork, int *info) {
  cunmqr_(&side, &trans, &m, &n, &k, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(tau), reinterpret_cast<std::complex<float>*>(c), &ldc, reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackOrmqr<double>(char side, char trans, int m, int n, int k, double *a, int lda, double *tau, double *c, int ldc, double *work, int lwork, int *info) {
  dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}

template<> void lapackOrmqr<float>(char side, char trans, int m, int n, int k, float *a, int lda, float *tau, float *c, int ldc, float *work, int lwork, int *info) {
  sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
}

template<> void lapackSymeig<c10::complex<double>, double>(char jobz, char uplo, int n, c10::complex<double> *a, int lda, double *w, c10::complex<double> *work, int lwork, double *rwork, int *info) {
  zheev_(&jobz, &uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, w, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, info);
}

template<> void lapackSymeig<c10::complex<float>, float>(char jobz, char uplo, int n, c10::complex<float> *a, int lda, float *w, c10::complex<float> *work, int lwork, float *rwork, int *info) {
  cheev_(&jobz, &uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, w, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, info);
}

template<> void lapackSymeig<double>(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, double* rwork, int *info) {
  (void)rwork;  // unused
  dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
}

template<> void lapackSymeig<float>(char jobz, char uplo, int n, float *a, int lda, float *w, float *work, int lwork, float* rwork, int *info) {
  (void)rwork;  // unused
  ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
}

template<> void lapackSyevd<c10::complex<double>, double>(char jobz, char uplo, int n, c10::complex<double> *a, int lda, double *w, c10::complex<double> *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  zheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, w, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<c10::complex<float>, float>(char jobz, char uplo, int n, c10::complex<float> *a, int lda, float *w, c10::complex<float> *work, int lwork, float *rwork, int lrwork, int *iwork, int liwork, int *info) {
  cheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, w, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template<> void lapackSyevd<double>(char jobz, char uplo, int n, double *a, int lda, double *w, double *work, int lwork, double *rwork, int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template<> void lapackSyevd<float>(char jobz, char uplo, int n, float *a, int lda, float *w, float *work, int lwork, float *rwork, int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template<> void lapackEig<double>(char jobvl, char jobvr, int n, double *a, int lda, double *w, double* vl, int ldvl, double *vr, int ldvr, double *work, int lwork, double *rwork, int *info) {
  // lapack [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  double *wr = w;
  double *wi = w + n;
  (void)rwork; // unused
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

template<> void lapackEig<float>(char jobvl, char jobvr, int n, float *a, int lda, float *w, float* vl, int ldvl, float *vr, int ldvr, float *work, int lwork, float *rwork, int *info) {
  // lapack [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  float *wr = w;
  float *wi = w + n;
  (void)rwork; // unused
  sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
}

template<> void lapackEig<c10::complex<double>, double>(char jobvl, char jobvr, int n, c10::complex<double> *a, int lda, c10::complex<double> *w, c10::complex<double> *vl, int ldvl, c10::complex<double> *vr, int ldvr, c10::complex<double> *work, int lwork, double *rwork, int *info) {
  zgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<double>*>(a), &lda,
         reinterpret_cast<std::complex<double>*>(w),
         reinterpret_cast<std::complex<double>*>(vl), &ldvl,
         reinterpret_cast<std::complex<double>*>(vr), &ldvr,
         reinterpret_cast<std::complex<double>*>(work), &lwork,
         rwork, info);
}

template<> void lapackEig<c10::complex<float>, float>(char jobvl, char jobvr, int n, c10::complex<float> *a, int lda, c10::complex<float> *w, c10::complex<float> *vl, int ldvl, c10::complex<float> *vr, int ldvr, c10::complex<float> *work, int lwork, float *rwork, int *info) {
  cgeev_(&jobvl, &jobvr, &n,
         reinterpret_cast<std::complex<float>*>(a), &lda,
         reinterpret_cast<std::complex<float>*>(w),
         reinterpret_cast<std::complex<float>*>(vl), &ldvl,
         reinterpret_cast<std::complex<float>*>(vr), &ldvr,
         reinterpret_cast<std::complex<float>*>(work), &lwork,
         rwork, info);
}

template<> void lapackSvd<c10::complex<double>, double>(char jobz, int m, int n, c10::complex<double> *a, int lda,
                                  double *s, c10::complex<double> *u, int ldu, c10::complex<double> *vt, int ldvt, c10::complex<double> *work, int lwork, double *rwork, int *iwork, int *info) {
  zgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<double>*>(a), &lda, s, reinterpret_cast<std::complex<double>*>(u), &ldu,
          reinterpret_cast<std::complex<double>*>(vt), &ldvt, reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, iwork, info);
}

template<> void lapackSvd<c10::complex<float>, float>(char jobz, int m, int n, c10::complex<float> *a, int lda,
                                 float *s, c10::complex<float> *u, int ldu, c10::complex<float> *vt, int ldvt, c10::complex<float> *work, int lwork, float *rwork, int *iwork, int *info) {
  cgesdd_(&jobz, &m, &n, reinterpret_cast<std::complex<float>*>(a), &lda, s, reinterpret_cast<std::complex<float>*>(u), &ldu,
          reinterpret_cast<std::complex<float>*>(vt), &ldvt, reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, iwork, info);
}

template<> void lapackSvd<double>(char jobz, int m, int n, double *a, int lda,
                                  double *s, double *u, int ldu, double *vt, int ldvt, double *work, int lwork, double *rwork, int *iwork, int *info) {
  dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template<> void lapackSvd<float>(char jobz, int m, int n, float *a, int lda,
                                 float *s, float *u, int ldu, float *vt, int ldvt, float *work, int lwork, float *rwork, int *iwork, int *info) {
  sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template<> void lapackLuSolve<c10::complex<double>>(char trans, int n, int nrhs, c10::complex<double> *a, int lda, int *ipiv, c10::complex<double> *b, int ldb, int *info) {
  zgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<double>*>(a), &lda, ipiv, reinterpret_cast<std::complex<double>*>(b), &ldb, info);
}

template<> void lapackLuSolve<c10::complex<float>>(char trans, int n, int nrhs, c10::complex<float> *a, int lda, int *ipiv, c10::complex<float> *b, int ldb, int *info) {
  cgetrs_(&trans, &n, &nrhs, reinterpret_cast<std::complex<float>*>(a), &lda, ipiv, reinterpret_cast<std::complex<float>*>(b), &ldb, info);
}

template<> void lapackLuSolve<double>(char trans, int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int *info) {
  dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackLuSolve<float>(char trans, int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int *info) {
  sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGels<c10::complex<double>>(
    char trans, int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    c10::complex<double> *work, int lwork, int *info) {
  zgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      reinterpret_cast<std::complex<double>*>(work), &lwork, info);
}

template<> void lapackGels<c10::complex<float>>(
    char trans, int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    c10::complex<float> *work, int lwork, int *info) {
  cgels_(&trans, &m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      reinterpret_cast<std::complex<float>*>(work), &lwork, info);
}

template<> void lapackGels<double>(
    char trans, int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *work, int lwork, int *info) {
  dgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

template<> void lapackGels<float>(
    char trans, int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *work, int lwork, int *info) {
  sgels_(&trans, &m, &n, &nrhs,
      a, &lda, b, &ldb, work, &lwork, info);
}

template<> void lapackGelsd<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *iwork, int *info) {
  zgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, iwork, info);
}

template<> void lapackGelsd<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *iwork, int *info) {
  cgelsd_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, iwork, info);
}

template<> void lapackGelsd<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *iwork, int *info) {
  dgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsd<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *iwork, int *info) {
  sgelsd_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, iwork, info);
}

template<> void lapackGelsy<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    int *jpvt, double rcond, int *rank,
    c10::complex<double> *work, int lwork, double *rwork, int *info) {
  zgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    int *jpvt, float rcond, int *rank,
    c10::complex<float> *work, int lwork, float *rwork, int *info) {
  cgelsy_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      jpvt, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelsy<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    int *jpvt, double rcond, int *rank,
    double *work, int lwork, double *rwork, int *info) {
  dgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelsy<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    int *jpvt, float rcond, int *rank,
    float *work, int lwork, float *rwork, int *info) {
  sgelsy_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      jpvt, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelss<c10::complex<double>, double>(
    int m, int n, int nrhs,
    c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb,
    double *s, double rcond, int *rank,
    c10::complex<double> *work, int lwork,
    double *rwork, int *info
    ) {
  zgelss_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<double>*>(a), &lda,
      reinterpret_cast<std::complex<double>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<double>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelss<c10::complex<float>, float>(
    int m, int n, int nrhs,
    c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb,
    float *s, float rcond, int *rank,
    c10::complex<float> *work, int lwork,
    float *rwork, int *info
    ) {
  cgelss_(&m, &n, &nrhs,
      reinterpret_cast<std::complex<float>*>(a), &lda,
      reinterpret_cast<std::complex<float>*>(b), &ldb,
      s, &rcond, rank,
      reinterpret_cast<std::complex<float>*>(work), &lwork,
      rwork, info);
}

template<> void lapackGelss<double>(
    int m, int n, int nrhs,
    double *a, int lda, double *b, int ldb,
    double *s, double rcond, int *rank,
    double *work, int lwork,
    double *rwork, int *info) {
  dgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}

template<> void lapackGelss<float>(
    int m, int n, int nrhs,
    float *a, int lda, float *b, int ldb,
    float *s, float rcond, int *rank,
    float *work, int lwork,
    float *rwork, int *info) {
  sgelss_(&m, &n, &nrhs,
      a, &lda, b, &ldb,
      s, &rcond, rank,
      work, &lwork, info);
}
#endif

// Below of the definitions of the functions operating on a batch that are going to be dispatched
// in the main helper functions for the linear algebra operations

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the solution to a system of linear equations
  A X = B,
where A is an n-by-n matrix and X and B are n-by-nrhs matrices.
Note that B is required to be a matrix, the usual, vector case, is obtained with nrhs = 1.
Above description is for non-batched input, the batched input is also supported.
This is an in-place routine, content of both A and b are overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see LAPACK's documentation for GESV routine.
*/
template<typename scalar_t>
static void apply_solve(Tensor& b, Tensor& A, Tensor& infos) {
#ifndef USE_LAPACK
  AT_ERROR("solve: LAPACK library not found in compilation");
#else
  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);
  auto lda = std::max<int64_t>(1, n);

  auto ipiv = at::empty({lda}, b.options().dtype(kInt));
  auto ipiv_data = ipiv.data_ptr<int>();
  auto infos_data = infos.data_ptr<int>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    int* info_working_ptr = &infos_data[i];
    lapackSolve<scalar_t>(n, nrhs, A_working_ptr, lda, ipiv_data, b_working_ptr, lda, info_working_ptr);
  }
#endif
}

std::tuple<Tensor, Tensor> _solve_helper_cpu(const Tensor& self, const Tensor& A) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  // infos might not get filled for empty inputs therefore at::zeros is used instead of at::empty
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(self))}, self.options().dtype(kInt));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "solve_cpu", [&]{
    apply_solve<scalar_t>(self_working_copy, A_working_copy, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "solve_cpu");
  } else {
    singleCheckErrors(infos.item().toInt(), "solve_cpu");
  }
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> solve(const Tensor& self, const Tensor& A) {
  TORCH_WARN_ONCE(
    "torch.solve is deprecated in favor of torch.linalg.solve",
    "and will be removed in a future PyTorch release.\n",
    "torch.linalg.solve has its arguments reversed and does not return the LU factorization.\n",
    "To get the LU factorization see torch.lu, which can be used with torch.lu_solve or torch.lu_unpack.\n",
    "X = torch.solve(B, A).solution\n",
    "should be replaced with\n",
    "X = torch.linalg.solve(A, B)"
  );
  TORCH_CHECK(self.dim() >= 2,
           "B should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "A should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = _linalg_broadcast_batch_dims(self, A, "solve");
  return at::_solve_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> solve_out(const Tensor& self, const Tensor& A, Tensor& solution, Tensor& lu) {
  TORCH_WARN_ONCE(
    "torch.solve is deprecated in favor of torch.linalg.solve",
    "and will be removed in a future PyTorch release.\n",
    "torch.linalg.solve has its arguments reversed and does not return the LU factorization.\n",
    "To get the LU factorization see torch.lu, which can be used with torch.lu_solve or torch.lu_unpack.\n",
    "X = torch.solve(B, A).solution\n",
    "should be replaced with\n",
    "X = torch.linalg.solve(A, B)"
  );
  checkSameDevice("solve", solution, self, "solution");
  checkSameDevice("solve", lu, self, "lu");
  checkLinalgCompatibleDtype("solve", solution, self, "solution");
  checkLinalgCompatibleDtype("solve", lu, self, "lu");

  Tensor solution_tmp, lu_tmp;
  std::tie(solution_tmp, lu_tmp) = at::_solve_helper(self, A);

  at::native::resize_output(solution, solution_tmp.sizes());
  at::native::resize_output(lu, lu_tmp.sizes());
  solution.copy_(solution_tmp);
  lu.copy_(lu_tmp);
  return std::tuple<Tensor&, Tensor&>(solution, lu);
}


// This is a type dispatching helper function for 'apply_solve'
Tensor& _linalg_solve_out_helper_cpu(Tensor& result, Tensor& input, Tensor& infos) {
  // 'result' and 'input' should be in column major order (it should be checked before calling this function)
  // the content of 'result', 'input' and 'infos' is overwritten by 'apply_solve'
  // 'result' should contain data of 'other' tensor (right-hand-side of the linear system of equations)
  // 'input' should contain data of original 'input' tensor (left-hand-side of the linear system of equations)
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "linalg_solve_out_cpu", [&]{
    apply_solve<scalar_t>(result, input, infos);
  });
  return result;
}

// Solves a system of linear equations matmul(input, x) = other in-place
// LAPACK/MAGMA error codes are saved in 'infos' tensor, they are not checked here
static Tensor& linalg_solve_out_info(Tensor& result, Tensor& infos, const Tensor& input, const Tensor& other) {
  checkSameDevice("linalg_solve", result, input);
  checkSameDevice("linalg_solve", other, input, "other");
  checkLinalgCompatibleDtype("linalg_solve", result, input);

  TORCH_CHECK(input.scalar_type() == other.scalar_type(),
    "input dtype ", input.scalar_type(), " does not match other dtype ", other.scalar_type());

  TORCH_CHECK(input.dim() >= 2,
           "input should have at least 2 dimensions, but has ", input.dim(), " dimensions instead");
  TORCH_CHECK(other.dim() >= 1,
           "other should have at least 1 dimension, but has ", other.dim(), " dimensions instead");

  // Two types of 'other' tensors are supported:
  // - 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  // - 2-dimensional (2D) tensor or batch of 2D tensors (matrix case)
  // original torch.solve supported only the matrix case, while NumPy works for both cases
  // for the batched input we need to be able to distinguish them
  bool vector_case = linalg_solve_is_vector_rhs(input, other);

  bool is_batched_column_major = false;
  if (vector_case) {
    is_batched_column_major = result.is_contiguous();
  } else if (!vector_case && result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if 'other' is a batch of 2D tensors, then 'input' can be non-batched and will be broadcasted
  auto expected_shape = IntArrayRef(input.sizes().data(), input.dim() - 1);  // input.shape[:-1]
  if (!vector_case && other.dim() > 2) {
    expected_shape = other.sizes();
  }

  bool result_equal_expected_shape = result.sizes().equals(expected_shape);
  bool result_input_same_type = (result.scalar_type() == input.scalar_type());

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;  // or result does not have the same dtype as input
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = linalg_solve_out_info(result_tmp, infos, input, other);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
    return result;
  }
  // else use result's storage directly

  // we need to unsqueeze 'other' because 2-dimensional tensors are expected in the implementation
  Tensor other_ = vector_case ? other.unsqueeze(-1) : other;

  // _linalg_broadcast_batch_dims also includes linearSolveCheckInputs
  // it checks for squareness of 'input' and 'shape' compatibility of 'other' and 'input'
  Tensor other_broadcasted, input_broadcasted;
  std::tie(other_broadcasted, input_broadcasted) = _linalg_broadcast_batch_dims(other_, input, "linalg_solve");

  auto squeezed_other_broadcasted = at::squeeze(other_broadcasted, -1);
  auto squeezed_result_shape = squeezed_other_broadcasted.sizes();

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    if (vector_case) {
      result.resize_(squeezed_result_shape);
    } else {
      at::native::resize_as_(result, other_broadcasted.transpose(-2, -1), MemoryFormat::Contiguous);
      result.transpose_(-2, -1);
    }
  }

  auto expected_result_shape = vector_case ? squeezed_result_shape : other_broadcasted.sizes();
  TORCH_INTERNAL_ASSERT(result.sizes().equals(expected_result_shape));
  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // result tensor must be in batched column major order (Fortran contiguous) for 2D inputs
  // or C contiguous for 1D input
  if (vector_case) {
    TORCH_INTERNAL_ASSERT(result.is_contiguous());
  } else {
    TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  }

  // for 1-dimensional 'other', we need to unsqueeze the result before passing to "apply_solve"
  if (vector_case) {
    result = result.unsqueeze_(-1);
  }

  // _linalg_solve_out_helper_ (apply_solve) performs calculations in-place and result must be a copy of other_broadcasted
  result.copy_(other_broadcasted);

  auto input_working_copy = cloneBatchedColumnMajor(input_broadcasted);

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == input.device());
  infos.resize_({std::max<int64_t>(1, batchCount(input_broadcasted))});
  // if input is empty infos might not get filled; make sure infos doesn't contain garbage then
  if (input.numel() == 0) {
    infos.fill_(0);
  }

  result = at::_linalg_solve_out_helper_(result, input_working_copy, infos);

  // for 1-dimensional 'other', we need to squeeze the result after "apply_solve"
  if (vector_case) {
    result = result.squeeze_(-1);
  }

  return result;
}

// Solves a system of linear equations matmul(input, x) = other in-place
Tensor& linalg_solve_out(const Tensor& input, const Tensor& other, Tensor& result) {
  auto infos = at::empty({0}, input.options().dtype(kInt));
  result = linalg_solve_out_info(result, infos, input, other);

  // Now check LAPACK/MAGMA error codes
  // batchCheckErrors(Tensor, char*) calls 'infos = infos.to(kCPU)'
  bool vector_case = linalg_solve_is_vector_rhs(input, other);
  if (vector_case ? result.dim() > 1 : result.dim() > 2) {
    batchCheckErrors(infos, "linalg_solve");
  } else {
    singleCheckErrors(infos.item().toInt(), "linalg_solve");
  }

  return result;
}

// Solves a system of linear equations matmul(input, x) = other
Tensor linalg_solve(const Tensor& input, const Tensor& other) {
  Tensor result = at::empty({0}, input.options());
  result = at::linalg_solve_out(result, input, other);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the inverse of n-by-n matrix 'self'
This is an in-place routine, it overwrites the content of 'self'.
'infos_lu' and 'infos_getri' are int Tensors containing error codes for each matrix in the batched input.
'infos_lu' is for holding lapackLU errors, and 'infos_getri' is for holding lapackGetri errors.
For more information see LAPACK's documentation for GETRI and GETRF routines.
*/
template <typename scalar_t>
static void apply_inverse(Tensor& self, Tensor& infos_lu, Tensor& infos_getri) {
#ifndef USE_LAPACK
  AT_ERROR("inverse: LAPACK library not found in compilation");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto batch_size = batchCount(self);
  auto n = self.size(-2);
  auto lda = std::max<int64_t>(1, n);

  auto ipiv = at::empty({lda}, self.options().dtype(kInt));
  auto ipiv_data = ipiv.data_ptr<int>();
  auto infos_lu_data = infos_lu.data_ptr<int>();
  auto infos_getri_data = infos_getri.data_ptr<int>();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  // Run once, first to get the optimum work size
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackGetri<scalar_t>(n, self_data, lda, ipiv_data, &wkopt, lwork, &info);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());
  auto work_data = work.data_ptr<scalar_t>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    int* info_lu_working_ptr = &infos_lu_data[i];
    lapackLu<scalar_t>(n, n, self_working_ptr, lda, ipiv_data, info_lu_working_ptr);

    // now compute the actual inverse
    int* info_getri_working_ptr = &infos_getri_data[i];
    lapackGetri<scalar_t>(n, self_working_ptr, lda, ipiv_data, work_data, lwork, info_getri_working_ptr);
  }
#endif
}

Tensor _inverse_helper_cpu(const Tensor& self) {
  auto infos_lu = at::empty({std::max<int64_t>(1, batchCount(self))}, self.options().dtype(kInt));
  auto infos_getri = at::empty({std::max<int64_t>(1, batchCount(self))}, self.options().dtype(kInt));
  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cpu", [&]{
    apply_inverse<scalar_t>(self_working_copy, infos_lu, infos_getri);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos_lu, "inverse_cpu");
    batchCheckErrors(infos_getri, "inverse_cpu");
  } else {
    singleCheckErrors(infos_lu.item().toInt(), "inverse_cpu");
    singleCheckErrors(infos_getri.item().toInt(), "inverse_cpu");
  }
  return self_working_copy;
}

Tensor inverse(const Tensor &self) {
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  squareCheckInputs(self);
  return at::_inverse_helper(self);
}

Tensor& inverse_out(const Tensor &self, Tensor &result) {
  checkSameDevice("inverse", result, self);
  checkLinalgCompatibleDtype("inverse", result, self);
  Tensor result_tmp = at::inverse(self);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

// This is a type dispatching helper function for 'apply_inverse'
Tensor& _linalg_inv_out_helper_cpu(Tensor &result, Tensor& infos_lu, Tensor& infos_getri) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_inverse'
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "linalg_inv_out_cpu", [&]{
    apply_inverse<scalar_t>(result, infos_lu, infos_getri);
  });
  return result;
}

// Computes the inverse matrix of 'input', it is is saved to 'result' in-place
// LAPACK/MAGMA/cuSOLVER error codes are saved in 'infos' tensors, they are not checked here
static Tensor& linalg_inv_out_info(Tensor& result, Tensor& infos_lu, Tensor& infos_getri, const Tensor& input) {
  squareCheckInputs(input);
  checkSameDevice("linalg_inv", result, input);
  checkLinalgCompatibleDtype("linalg_inv", result, input);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_lu.scalar_type() == kInt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_getri.scalar_type() == kInt);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_lu.device() == input.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_getri.device() == input.device());

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;  // or result does not have the same dtype as input
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor

  // similar conditions for infos_lu and infos_getri tensors
  auto expected_info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  copy_needed |= (infos_lu.numel() != 0 && !infos_lu.is_contiguous());
  copy_needed |= (infos_lu.numel() != 0 && !(infos_lu.sizes().equals(expected_info_shape)));

  copy_needed |= (infos_getri.numel() != 0 && !infos_getri.is_contiguous());
  copy_needed |= (infos_getri.numel() != 0 && !(infos_getri.sizes().equals(expected_info_shape)));

  if (copy_needed) {
    Tensor result_tmp = at::empty(input.sizes(), input.options());
    result_tmp.transpose_(-2, -1);
    Tensor infos_lu_tmp = at::zeros({expected_info_shape}, input.options().dtype(kInt));
    Tensor infos_getri_tmp = at::zeros({expected_info_shape}, input.options().dtype(kInt));

    result_tmp = linalg_inv_out_info(result_tmp, infos_lu_tmp, infos_getri_tmp, input);

    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
    at::native::resize_output(infos_lu, infos_lu_tmp.sizes());
    infos_lu.copy_(infos_lu_tmp);
    at::native::resize_output(infos_getri, infos_getri_tmp.sizes());
    infos_getri.copy_(infos_getri_tmp);
    return result;
  }
  // else  use result's storage directly

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.transpose(-2, -1), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(input.sizes()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.device() == input.device());

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.transpose(-2, -1).is_contiguous());

  // if info has no elements we can modify it
  if (infos_lu.numel() == 0) {
    infos_lu.resize_(expected_info_shape);
    infos_lu.fill_(0);
  }
  if (infos_getri.numel() == 0) {
    infos_getri.resize_(expected_info_shape);
    infos_getri.fill_(0);
  }

  // info tensors must be contiguous
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_lu.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_lu.sizes().equals(expected_info_shape));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_getri.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos_getri.sizes().equals(expected_info_shape));

  // _linalg_inv_out_helper_ (apply_inverse) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  // TODO: Replace this helper with DECLARE/DEFINE_DISPATCH
  result = at::_linalg_inv_out_helper_(result, infos_lu, infos_getri);
  return result;
}

// Computes the inverse matrix of 'input', it is is saved to 'result' in-place
Tensor& linalg_inv_out(const Tensor &input, Tensor &result) {
  auto info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  auto infos_lu = at::zeros({info_shape}, input.options().dtype(kInt));
  auto infos_getri = at::zeros({info_shape}, input.options().dtype(kInt));
  result = linalg_inv_out_info(result, infos_lu, infos_getri, input);

  // Now check LAPACK/MAGMA/cuSOLVER error codes
  if (result.dim() > 2) {
    batchCheckErrors(infos_lu, "linalg_inv_lu");
    batchCheckErrors(infos_getri, "linalg_inv_getri");
  } else {
    singleCheckErrors(infos_lu.item().toInt(), "linalg_inv_lu");
    singleCheckErrors(infos_getri.item().toInt(), "linalg_inv_getri");
  }

  return result;
}

// Computes the inverse matrix of 'input'
Tensor linalg_inv(const Tensor &input) {
  Tensor result, info;
  std::tie(result, info) = at::linalg_inv_ex(input, /*check_errors=*/false);

  // we pass check_errors=false above and do the check here
  // so that the name of the function is correct in the error message
  if (input.dim() > 2) {
    batchCheckErrors(info, "torch.linalg.inv");
  } else {
    singleCheckErrors(info.item<int64_t>(), "torch.linalg.inv");
  }

  return result;
}

std::tuple<Tensor&, Tensor&> linalg_inv_ex_out(const Tensor& input, bool check_errors, Tensor& inverse, Tensor& info) {
  squareCheckInputs(input);
  ScalarType info_output_type = ScalarType::Int;
  TORCH_CHECK(
      info.scalar_type() == info_output_type,
      "torch.linalg.inv_ex: ",
      "Expected info to have ", info_output_type, " dtype, but got info with dtype ", info.scalar_type());

  // provided `info` tensor is used to save the information about the LU decomposition of `input`
  // in addition current implementation requires a separate tensor
  // for saving the information about the inversion process after the LU decomposition
  auto expected_info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  auto info_inversion = at::zeros({expected_info_shape}, input.options().dtype(kInt));

  linalg_inv_out_info(inverse, info, info_inversion, input);

  if (check_errors) {
    if (input.dim() > 2) {
      batchCheckErrors(info, "torch.linalg.inv_ex");
    } else {
      singleCheckErrors(info.item().toInt(), "torch.linalg.inv_ex");
    }
  }

  return std::tuple<Tensor&, Tensor&>(inverse, info);
}

std::tuple<Tensor, Tensor> linalg_inv_ex(const Tensor& input, bool check_errors) {
  squareCheckInputs(input);
  Tensor inverse = at::empty(input.sizes(), input.options(), MemoryFormat::Contiguous);
  inverse.transpose_(-2, -1); // make `inverse` tensor with batched column major format
  auto info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  Tensor info = at::zeros({info_shape}, input.options().dtype(kInt));
  std::tie(inverse, info) = at::native::linalg_inv_ex_out(input, check_errors, inverse, info);
  return std::make_tuple(inverse, info);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("cholesky_solve: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    lapackCholeskySolve<scalar_t>(uplo, n, nrhs, A_working_ptr, n, b_working_ptr, n, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

Tensor _cholesky_solve_helper_cpu(const Tensor& self, const Tensor& A, bool upper) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  std::vector<int64_t> infos(batchCount(self), 0);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cpu", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "cholesky_solve_cpu");
  } else {
    singleCheckErrors(infos[0], "cholesky_solve_cpu");
  }
  return self_working_copy;
}

// Supports arbitrary batch dimensions for self and A
Tensor cholesky_solve(const Tensor& self, const Tensor& A, bool upper) {
  TORCH_CHECK(self.dim() >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = _linalg_broadcast_batch_dims(self, A, "cholesky_solve");
  return at::_cholesky_solve_helper(self_broadcasted, A_broadcasted, upper);
}

Tensor& cholesky_solve_out(const Tensor& self, const Tensor& A, bool upper, Tensor& result) {
  checkSameDevice("cholesky_solve", result, self);
  checkLinalgCompatibleDtype("cholesky_solve", result, self);
  Tensor result_tmp = at::cholesky_solve(self, A, upper);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(cholesky_stub);

Tensor cholesky(const Tensor &self, bool upper) {
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",
    "L = torch.cholesky(A)\n",
    "should be replaced with\n",
    "L = torch.linalg.cholesky(A)\n",
    "and\n"
    "U = torch.cholesky(A, upper=True)\n",
    "should be replaced with\n",
    "U = torch.linalg.cholesky(A.transpose(-2, -1).conj()).transpose(-2, -1).conj()"
  );
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  squareCheckInputs(self);

  auto raw_cholesky_output = cloneBatchedColumnMajor(self);
  auto info_shape = IntArrayRef(
      self.sizes().cbegin(), self.sizes().cend() - 2); // self.shape[:-2]
  auto info = at::empty({info_shape}, self.options().dtype(kInt));

  // fill the raw_cholesky_output with the result
  cholesky_stub(self.device().type(), raw_cholesky_output, info, upper);

  if (self.dim() > 2) {
    batchCheckErrors(info, "cholesky");
  } else {
    singleCheckErrors(info.item<int64_t>(), "cholesky");
  }

  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

Tensor& cholesky_out(const Tensor &self, bool upper, Tensor &result) {
   TORCH_WARN_ONCE(
    "torch.cholesky is deprecated in favor of torch.linalg.cholesky and will be ",
    "removed in a future PyTorch release.\n",
    "L = torch.cholesky(A)\n",
    "should be replaced with\n",
    "L = torch.linalg.cholesky(A)\n",
    "and\n"
    "U = torch.cholesky(A, upper=True)\n",
    "should be replaced with\n",
    "U = torch.linalg.cholesky(A.transpose(-2, -1).conj()).transpose(-2, -1).conj()"
  );
  checkSameDevice("cholesky", result, self);
  checkLinalgCompatibleDtype("cholesky", result, self);
  Tensor result_tmp = at::cholesky(self, upper);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

void linalg_cholesky_out_info(const Tensor& input, const Tensor& result, const Tensor& info) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-1) == input.size(-2));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.device() == input.device());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.device() == input.device());

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.transpose(-2, -1), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(input.sizes()));

  // cholesky_stub (apply_cholesky) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  // if info has no elements we can modify it
  auto expected_info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  if (info.numel() == 0) {
    info.resize_(expected_info_shape);
  }

  // info must be contiguous
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.sizes().equals(expected_info_shape));
  info.fill_(0);

  cholesky_stub(result.device().type(), result, info, /*upper=*/false);

  result.tril_();
}

std::tuple<Tensor&, Tensor&> linalg_cholesky_ex_out(const Tensor& input, bool check_errors, Tensor& L, Tensor& info) {
  squareCheckInputs(input);
  checkSameDevice("torch.linalg.cholesky_ex", L, input, "L");
  checkLinalgCompatibleDtype("torch.linalg.cholesky_ex", L, input, "L");
  checkSameDevice("torch.linalg.cholesky_ex", info, input, "info");

  // Do not allow type promotion for the `info` tensor, it must be of Int dtype
  // Int is used because current interface to LAPACK and its CUDA implementation use "int" type.
  // https://github.com/pytorch/pytorch/pull/56724#discussion_r618916774
  ScalarType info_output_type = ScalarType::Int;
  TORCH_CHECK(
      info.scalar_type() == info_output_type,
      "torch.linalg.cholesky_ex: ",
      "Expected info to have ", info_output_type, " dtype, but got info with dtype ", info.scalar_type());

  bool L_input_same_type = (L.scalar_type() == input.scalar_type());
  bool L_equal_expected_shape = L.sizes().equals(input.sizes());
  bool is_L_batched_column_major = false;
  if (L.dim() >= 2) {
    is_L_batched_column_major = L.transpose(-2, -1).is_contiguous();
  }

  // if L is not empty and not in batched column major format
  bool copy_needed = (L.numel() != 0 && !is_L_batched_column_major);
  copy_needed |= (L.numel() != 0 && !L_equal_expected_shape); // or L does not have the expected shape
  copy_needed |= !L_input_same_type;  // or L does not have the same dtype as input
  // we have to allocate a temporary tensor

  // similar conditions for info tensor
  auto expected_info_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2); // input.shape[:-2]
  copy_needed |= (info.numel() != 0 && !info.is_contiguous());
  copy_needed |= (info.numel() != 0 && !(info.sizes().equals(expected_info_shape))); // or L does not have the expected shape

  if (copy_needed) {
    Tensor L_tmp = at::empty({0}, input.options());
    Tensor info_tmp = at::empty({0}, input.options().dtype(kInt));
    linalg_cholesky_out_info(input, L_tmp, info_tmp);
    at::native::resize_output(L, L_tmp.sizes());
    L.copy_(L_tmp);
    at::native::resize_output(info, info_tmp.sizes());
    info.copy_(info_tmp);
  } else {
    // use "out" tensors' memory directly
    linalg_cholesky_out_info(input, L, info);
  }

  if (check_errors) {
    if (input.dim() > 2) {
      batchCheckErrors(info, "torch.linalg.cholesky_ex");
    } else {
      singleCheckErrors(info.item<int64_t>(), "torch.linalg.cholesky_ex");
    }
  }

  return std::tuple<Tensor&, Tensor&>(L, info);
}

std::tuple<Tensor, Tensor> linalg_cholesky_ex(const Tensor& input, bool check_errors) {
  Tensor L = at::empty({0}, input.options());
  Tensor info = at::empty({0}, input.options().dtype(kInt));
  std::tie(L, info) = at::native::linalg_cholesky_ex_out(input, check_errors, L, info);
  return std::make_tuple(L, info);
}

Tensor linalg_cholesky(const Tensor &self) {
  Tensor result, info;
  std::tie(result, info) = at::linalg_cholesky_ex(self, /*check_errors=*/false);

  // we pass check_errors=false above and do the check here
  // so that the name of the function is correct in the error message
  if (self.dim() > 2) {
    batchCheckErrors(info, "torch.linalg.cholesky");
  } else {
    singleCheckErrors(info.item<int64_t>(), "torch.linalg.cholesky");
  }

  return result;
}

Tensor& linalg_cholesky_out(const Tensor &self, Tensor &result) {
  // linalg_cholesky_ex_outf includes these checks, but we do it here
  // so that the name of the function is correct in the error message
  checkSameDevice("torch.linalg.cholesky", result, self);
  checkLinalgCompatibleDtype("torch.linalg.cholesky", result, self);

  Tensor info = at::empty({0}, self.options().dtype(kInt));
  std::tie(result, info) = at::linalg_cholesky_ex_outf(self, /*check_errors=*/false, result, info);

  // we pass check_errors=false above and do the check here
  // so that the name of the function is correct in the error message
  if (self.dim() > 2) {
    batchCheckErrors(info, "torch.linalg.cholesky");
  } else {
    singleCheckErrors(info.item<int64_t>(), "torch.linalg.cholesky");
  }

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cholesky_inverse_stub);

Tensor& cholesky_inverse_out_info(Tensor& result, Tensor& infos, const Tensor& input, bool upper) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-1) == input.size(-2));

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.transpose(-2, -1), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // cholesky_inverse_stub (apply_cholesky_inverse) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  // infos must be contiguous
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  infos.fill_(0);

  result = cholesky_inverse_stub(result.device().type(), result, infos, upper);
  return result;
}

Tensor& cholesky_inverse_out(const Tensor &input, bool upper, Tensor &result) {
  squareCheckInputs(input);
  checkSameDevice("cholesky_inverse", result, input);
  checkLinalgCompatibleDtype("cholesky_inverse", result, input);

  // MAGMA requires 'infos' to reside in CPU memory, therefore we create 'infos' only on CPU for now.
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt).device(kCPU));

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;  // or result does not have the same dtype as input
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = cholesky_inverse_out_info(result_tmp, infos, input, upper);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // use result's memory directly
    result = cholesky_inverse_out_info(result, infos, input, upper);
  }

  // Now check LAPACK/MAGMA error codes
  if (result.dim() > 2) {
    batchCheckErrors(infos, "cholesky_inverse");
  } else {
    singleCheckErrors(infos.item().toInt(), "cholesky_inverse");
  }
  return result;
}

Tensor cholesky_inverse(const Tensor &input, bool upper) {
  Tensor result = at::empty({0}, input.options());
  result = at::cholesky_inverse_out(result, input, upper);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lu_stub);

std::tuple<Tensor, Tensor, Tensor> _lu_with_info(const Tensor& self, bool compute_pivots, bool check_errors) {
  TORCH_CHECK(self.dim() >= 2,
           "expected tensor with 2 or more dimensions, got size: ", self.sizes(),
           " instead");
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size.back() = std::min(m, n);
  auto pivots_tensor = at::empty(req_size, self.options().dtype(kInt));
  req_size.pop_back();
  auto infos_tensor = at::zeros(req_size, self.options().dtype(kInt));

  // lu_stub (apply_lu) requires batched column major (Fortran-contiguous) tensors
  // 'lu' tensor is modified in-place and must be a copy of 'self'
  Tensor lu = cloneBatchedColumnMajor(self);
  lu_stub(self.device().type(), lu, pivots_tensor, infos_tensor, compute_pivots);

  if (check_errors) {
    if (self.dim() > 2) {
      batchCheckErrors(infos_tensor, "lu", /*allow_singular=*/true);
    } else {
      singleCheckErrors(infos_tensor.item<int64_t>(), "lu", /*allow_singular=*/true);
    }
  }
  return std::make_tuple(lu, pivots_tensor, infos_tensor);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(triangular_solve_stub);

/*
Solves the matrix equation 'input' @ 'result' = 'other' for the 'result'.
The result of the computation is saved in-place in 'result' tensor,
'clone_input' will be a copy of 'input',
'infos' is used to store information for possible checks for error,
'upper' controls the portion of input matrix to consider in computations,
'transpose' if true then 'input.transpose(-2, -1)' @ 'result' = 'other' is solved,
'unitriangular' if true then the diagonal elements of 'input' are assumed to be 1
and the actual diagonal values are not used.
*/
static std::tuple<Tensor&, Tensor&> triangular_solve_out_info(
    Tensor& result,
    Tensor& clone_input,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other,
    bool upper, bool transpose, bool unitriangular) {
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT(input.device() == other.device());
  TORCH_INTERNAL_ASSERT(input.device() == result.device());
  TORCH_INTERNAL_ASSERT(input.device() == clone_input.device());
  TORCH_INTERNAL_ASSERT(input.device() == infos.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT(input.scalar_type() == result.scalar_type());
  TORCH_INTERNAL_ASSERT(input.scalar_type() == clone_input.scalar_type());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());

  // if 'result' has no elements we can modify it
  if (result.numel() == 0) {
    result.resize_(other.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);  // make 'result' to have Fortran contiguous memory layout
  }

  // if 'clone_input' has no elements we can modify it
  if (clone_input.numel() == 0) {
    clone_input.resize_(input.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
    clone_input.transpose_(-2, -1);  // make 'clone_input' to have Fortran contiguous memory layout
  }

  // 'result' and 'clone_input' must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT(clone_input.transpose(-2, -1).is_contiguous());

  // triangular_solve_stub performs calculations in-place
  // 'result' must be a copy of 'other'
  // 'clone_input' must be a copy of 'input'
  TORCH_INTERNAL_ASSERT(result.sizes().equals(other.sizes()));
  TORCH_INTERNAL_ASSERT(clone_input.sizes().equals(input.sizes()));
  result.copy_(other);
  clone_input.copy_(input);

  triangular_solve_stub(input.device().type(), clone_input, result, infos, upper, transpose, /*conjugate_transpose=*/false, unitriangular);

  return std::tuple<Tensor&, Tensor&>(result, clone_input);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor, Tensor> triangular_solve(const Tensor& self, const Tensor& A,
                                            bool upper, bool transpose, bool unitriangular) {
  TORCH_CHECK(self.dim() >= 2,
           "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ", A.dim(), " dimensions instead");

  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = _linalg_broadcast_batch_dims(self, A, "triangular_solve");

  Tensor result = at::empty({0}, self.options());
  Tensor clone_A = at::empty({0}, self.options());
  Tensor infos = at::zeros({std::max<int64_t>(1, batchCount(self_broadcasted))}, self.options().dtype(kInt));

  triangular_solve_out_info(result, clone_A, infos, A_broadcasted, self_broadcasted, upper, transpose, unitriangular);

  if (self_broadcasted.dim() > 2) {
    batchCheckErrors(infos, "triangular_solve");
  } else {
    singleCheckErrors(infos.item().toInt(), "triangular_solve");
  }

  return std::tuple<Tensor, Tensor>(result, clone_A);
}

std::tuple<Tensor&, Tensor&> triangular_solve_out(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular, Tensor& result, Tensor& clone_A) {
  checkSameDevice("triangular_solve", result, self);
  checkLinalgCompatibleDtype("triangular_solve", result, self);
  checkSameDevice("triangular_solve", clone_A, self, "clone_A");
  checkLinalgCompatibleDtype("triangular_solve", clone_A, self, "clone_A");
  Tensor result_tmp, clone_A_tmp;
  std::tie(result_tmp, clone_A_tmp) = at::native::triangular_solve(self, A, upper, transpose, unitriangular);
  at::native::resize_output(result, result_tmp.sizes());
  at::native::resize_output(clone_A, clone_A_tmp.sizes());
  result.copy_(result_tmp);
  clone_A.copy_(clone_A_tmp);
  return std::tuple<Tensor&, Tensor&>(result, clone_A);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(geqrf_stub);

static void geqrf_out_helper(const Tensor& input, const Tensor& QR, const Tensor& tau) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == QR.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == QR.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  // if 'QR' has no elements we can modify it
  if (QR.numel() == 0) {
    QR.resize_as_(input.transpose(-2, -1), MemoryFormat::Contiguous);
    QR.transpose_(-2, -1); // make Fortran-contiguous
  }

  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  if (tau.numel() == 0) {
    tau.resize_(expected_batch_tau_shape);
  }

  // QR tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(QR.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT(QR.sizes().equals(input.sizes()));

  // tau tensor must be contiguous
  TORCH_INTERNAL_ASSERT(tau.is_contiguous());
  TORCH_INTERNAL_ASSERT(tau.sizes().equals(expected_batch_tau_shape));

  // geqrf_stub (apply_geqrf) performs calculations in-place and 'QR' must be a copy of input
  QR.copy_(input);
  geqrf_stub(input.device().type(), QR, tau);
}

std::tuple<Tensor&, Tensor&> geqrf_out(const Tensor& input, Tensor& QR, Tensor& tau) {
  TORCH_CHECK(input.dim() >= 2, "torch.geqrf: input must have at least 2 dimensions.");

  checkSameDevice("torch.geqrf", QR, input, "a"); // 'a' is used in documentation and native_functions.yml
  checkSameDevice("torch.geqrf", tau, input, "tau");
  checkLinalgCompatibleDtype("torch.geqrf", QR, input, "a");
  checkLinalgCompatibleDtype("torch.geqrf", tau, input, "tau");

  bool QR_input_same_type = (QR.scalar_type() == input.scalar_type());
  bool tau_input_same_type = (tau.scalar_type() == input.scalar_type());
  bool QR_equal_expected_shape = QR.sizes().equals(input.sizes());

  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  bool tau_equal_expected_shape = tau.sizes().equals(expected_batch_tau_shape);

  bool is_batched_column_major = false;
  if (QR.dim() >= 2) {
    is_batched_column_major = QR.transpose(-2, -1).is_contiguous();
  }

  // if 'QR' is not empty and not in batched column major format
  bool copy_needed = (QR.numel() != 0 && !is_batched_column_major);
  copy_needed |= (QR.numel() != 0 && !QR_equal_expected_shape); // or 'QR' does not have the expected shape
  copy_needed |= !QR_input_same_type;  // or 'QR' does not have the same dtype as input
  // we have to allocate a temporary tensor

  copy_needed |= (tau.numel() != 0 && !tau.is_contiguous());
  copy_needed |= (tau.numel() != 0 && !tau_equal_expected_shape); // or 'tau' does not have the expected shape
  copy_needed |= !tau_input_same_type;  // or 'tau' does not have the same dtype as input

  if (copy_needed) {
    Tensor QR_tmp = at::empty({0}, input.options());
    Tensor tau_tmp = at::empty({0}, input.options());

    geqrf_out_helper(input, QR_tmp, tau_tmp);

    at::native::resize_output(QR, QR_tmp.sizes());
    QR.copy_(QR_tmp);
    at::native::resize_output(tau, tau_tmp.sizes());
    tau.copy_(tau_tmp);
  } else {
    // use "out" tensors' storage directly
    geqrf_out_helper(input, QR, tau);
  }

  return std::tuple<Tensor&, Tensor&>(QR, tau);
}

std::tuple<Tensor, Tensor> geqrf(const Tensor& input) {
  Tensor QR = at::empty({0}, input.options());
  Tensor tau = at::empty({0}, input.options());
  std::tie(QR, tau) = at::geqrf_outf(input, QR, tau);
  return std::make_tuple(QR, tau);
}

/*
  Computes the QR decomposition using GEQRF and ORGQR operations.
  This is an in-place function and Q, R tensors must have correct shape and be Fortran contiguous.

  Args:
  * `input` - [in] Input tensor for QR decomposition
  * `Q` - [out] Tensor containing the Q matrices of QR decomposition
  * `R` - [out] Tensor containing the R matrices of QR decomposition
  * `compute_q` - controls whether the Q tensor is computed
  * `reduced_mode` - controls the size of Q and R tensors

  For further details, please see the LAPACK documentation for GEQRF and ORGQR.
*/
void linalg_qr_out_helper(const Tensor& input, const Tensor& Q, const Tensor& R, bool compute_q, bool reduced_mode) {

  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == Q.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == Q.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == R.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == R.device());

  auto m = input.size(-2);
  auto n = input.size(-1);
  auto mn = std::min(m, n);

  // Q must have the expected shape: reduced_mode ? (..., m, min(m, n)) : (..., m, m)
  if (compute_q) {
    auto expected_Q_shape = input.sizes().vec();
    expected_Q_shape.back() = reduced_mode ? mn : m;
    TORCH_INTERNAL_ASSERT(Q.sizes().equals(expected_Q_shape));

    // Q tensor must be in batched column major order (Fortran contiguous)
    TORCH_INTERNAL_ASSERT(Q.transpose(-2, -1).is_contiguous());
  }

  // R must have the expected shape: (reduced_mode || !compute_q) ? (..., min(m,n), n) : (..., m, n)
  auto expected_R_shape = input.sizes().vec();
  expected_R_shape.end()[-2] = (reduced_mode || !compute_q) ? mn : m;
  TORCH_INTERNAL_ASSERT(R.sizes().equals(expected_R_shape));

  // R tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(R.transpose(-2, -1).is_contiguous());

  auto tau_shape = input.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  Tensor tau = at::empty(tau_shape, input.options());

  // geqrf requires m x n workspace input that is modified in-place
  // if m > n and reduced==true we use Q tensor for storing the result of geqrf operation
  // otherwise R tensor is used
  Tensor QR;
  if (m <= n) {
    QR = R;
  } else { // m > n
    if (compute_q) {
      QR = reduced_mode ? Q : R;
    } else {
      // if m > n and compute_q==false we need to allocate an additional temporary tensor
      QR = at::empty(input.transpose(-2, -1).sizes(), input.options());
      QR.transpose_(-2, -1);
    }
  }

  // geqrf_stub (apply_geqrf) performs calculations in-place and 'QR' must be a copy of input
  QR.copy_(input);
  geqrf_stub(input.device().type(), QR, tau);

  // this is for mode='r'
  if (!compute_q) {
    // if m > n we used a temporary tensor to store the result of geqrf
    if (m > n) {
      R.copy_(QR.slice(-2, 0, mn));
    }
    R.triu_();
    return;
  }

  // if Q tensor was used for geqrf copy the result for R from QR
  if (m > n && reduced_mode) {
    R.copy_(Q.slice(-2, 0, n));
  } else {
    Q.slice(-1, 0, n).copy_(R.slice(-1, 0, m));
  }
  R.triu_();

  // Next perform ORGQR for Q using the result from GEQRF
  orgqr_stub(input.device().type(), const_cast<Tensor&>(Q), tau);
}

std::tuple<Tensor, Tensor> _linalg_qr_helper_default(const Tensor& input, std::string mode) {
  bool compute_q, reduced_mode;
  std::tie(compute_q, reduced_mode) = _parse_qr_mode(mode);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto mn = std::min(m, n);

  // Allocate Q, R tensors with correct shape and memory layout
  Tensor Q;
  if (compute_q) {
    auto Qt_shape = input.sizes().vec();
    Qt_shape.end()[-2] = reduced_mode ? mn : m;
    Qt_shape.end()[-1] = m;
    Q = at::empty(Qt_shape, input.options());
    Q.transpose_(-2, -1); // make 'Q' with Fortran contiguous memory layout
  } else {
    Q = at::empty({0}, input.options());
  }

  auto Rt_shape = input.sizes().vec();
  Rt_shape.end()[-2] = n;
  Rt_shape.end()[-1] = (reduced_mode || !compute_q) ? mn : m;
  Tensor R = at::empty(Rt_shape, input.options());
  R.transpose_(-2, -1); // make 'R' with Fortran contiguous memory layout

  // Now fill Q, R tensors with the result
  linalg_qr_out_helper(input, Q, R, compute_q, reduced_mode);

  return std::make_tuple(Q, R);
}

std::tuple<Tensor,Tensor> linalg_qr(const Tensor& self, std::string mode) {
  TORCH_CHECK(self.dim() >= 2,
              "qr input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  return at::_linalg_qr_helper(self, mode);
}

std::tuple<Tensor&,Tensor&> linalg_qr_out(const Tensor& self, std::string mode, Tensor& Q, Tensor& R) {
  TORCH_CHECK(self.dim() >= 2,
              "torch.linalg.qr: input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  checkSameDevice("torch.linalg.qr", Q, self, "Q");
  checkSameDevice("torch.linalg.qr", R, self, "R");
  checkLinalgCompatibleDtype("torch.linalg.qr", Q, self, "Q");
  checkLinalgCompatibleDtype("torch.linalg.qr", R, self, "R");
  Tensor Q_tmp, R_tmp;
  std::tie(Q_tmp, R_tmp) = at::_linalg_qr_helper(self, mode);
  at::native::resize_output(Q, Q_tmp.sizes());
  Q.copy_(Q_tmp);
  at::native::resize_output(R, R_tmp.sizes());
  R.copy_(R_tmp);
  return std::tuple<Tensor&, Tensor&>(Q, R);
}

std::tuple<Tensor,Tensor> qr(const Tensor& self, bool some) {
  TORCH_WARN_ONCE(
    "torch.qr is deprecated in favor of torch.linalg.qr and will be removed in a future PyTorch release.\n",
    "The boolean parameter 'some' has been replaced with a string parameter 'mode'.\n",
    "Q, R = torch.qr(A, some)\n",
    "should be replaced with\n",
    "Q, R = torch.linalg.qr(A, 'reduced' if some else 'complete')"
  );
  std::string mode = some ? "reduced" : "complete";
  return at::linalg_qr(self, mode);
}

std::tuple<Tensor&,Tensor&> qr_out(const Tensor& self, bool some, Tensor& Q, Tensor& R) {
  TORCH_WARN_ONCE(
    "torch.qr is deprecated in favor of torch.linalg.qr and will be removed in a future PyTorch release.\n",
    "The boolean parameter 'some' has been replaced with a string parameter 'mode'.\n",
    "Q, R = torch.qr(A, some)\n",
    "should be replaced with\n",
    "Q, R = torch.linalg.qr(A, 'reduced' if some else 'complete')"
  );
  std::string mode = some ? "reduced" : "complete";
  return at::linalg_qr_out(Q, R, self, mode);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(orgqr_stub);

/*
  The householder_product (orgqr) function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as is produced by the geqrf function.

  Args:
  * `input` - Tensor with the directions of the elementary reflectors below the diagonal.
  * `tau` - Tensor containing the magnitudes of the elementary reflectors.
  * `result` - result Tensor, which will contain the orthogonal (or unitary) matrix Q.

  For further details, please see the LAPACK/MAGMA documentation.
*/
Tensor& householder_product_out_helper(const Tensor& input, const Tensor& tau, Tensor& result) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-2) >= input.size(-1));
  TORCH_INTERNAL_ASSERT(input.size(-1) >= tau.size(-1));

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.transpose(-2, -1), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT(result.sizes().equals(input.sizes()));

  // tau tensor must be contiguous
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // orgqr_stub (apply_orgqr) performs calculations in-place and result must be a copy of input
  result.copy_(input);

  result = orgqr_stub(result.device().type(), result, tau_);
  return result;
}

Tensor& linalg_householder_product_out(const Tensor& input, const Tensor& tau, Tensor& result) {
  TORCH_CHECK(input.dim() >= 2, "torch.linalg.householder_product: input must have at least 2 dimensions.");
  TORCH_CHECK(
      input.size(-2) >= input.size(-1),
      "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]");
  TORCH_CHECK(
      input.size(-1) >= tau.size(-1),
      "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]");

  TORCH_CHECK(
      input.dim() - tau.dim() == 1,
      "torch.linalg.householder_product: Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      tau.dim(),
      " and input.ndim is equal to ",
      input.dim());
  if (input.dim() > 2) {
    auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2); // input.shape[:-2]
    auto actual_batch_tau_shape = IntArrayRef(tau.sizes().data(), tau.dim() - 1); // tau.shape[:-1]
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_tau_shape),
        "torch.linalg.householder_product: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);
  }

  TORCH_CHECK(
      tau.scalar_type() == input.scalar_type(),
      "torch.linalg.householder_product: tau dtype ",
      tau.scalar_type(),
      " does not match input dtype ",
      input.scalar_type());
  TORCH_CHECK(
      input.device() == tau.device(),
      "torch.linalg.householder_product: Expected input and tau to be on the same device, but found input on ",
      input.device(),
      " and tau on ",
      tau.device(),
      " instead.");

  checkSameDevice("torch.linalg.householder_product", result, input);
  checkLinalgCompatibleDtype("torch.linalg.householder_product", result, input);

  // TODO: uncomment the following when passing incorrectly sized 'result' is not allowed
  // if (result.numel() != 0) {
  //   // Resize messes up the strides, so let's not use at::native::resize_output
  //   TORCH_CHECK(result.sizes().equals(input.sizes()),
  //   "result shape ", result.sizes(), " does not match input shape ", input.sizes());
  // }

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;  // or result does not have the same dtype as input
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = householder_product_out_helper(input, tau, result_tmp);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // use result's storage directly
    result = householder_product_out_helper(input, tau, result);
  }

  return result;
}

Tensor linalg_householder_product(const Tensor& input, const Tensor& tau) {
  Tensor result = at::empty({0}, input.options());
  result = at::linalg_householder_product_outf(input, tau, result);
  return result;
}

// torch.orgqr is an alias of torch.linalg.householder_product
// torch.linalg.householder_product is the preferred new function
Tensor& orgqr_out(const Tensor& input, const Tensor& tau, Tensor& result) {
  return at::linalg_householder_product_outf(input, tau, result);
}

Tensor orgqr(const Tensor& input, const Tensor& tau) {
  return at::linalg_householder_product(input, tau);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ormqr_stub);

void ormqr_out_helper(const Tensor& input, const Tensor& tau, const Tensor& other, const Tensor& result, bool left, bool transpose) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.dim() >= 2);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.size(left ? -2 : -1) >= tau.size(-1));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(other.size(left ? -2 : -1) == input.size(-2));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == tau.device());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == other.device());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.device() == input.device());

  // if 'result' has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, other.transpose(-2, -1), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // 'result' tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));

  // 'tau' tensor must be contiguous
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // 'input' tensor must be Fortran contiguous
  Tensor input_ = input;
  if (!input.transpose(-2, -1).is_contiguous()) {
    input_ = at::empty(input.transpose(-2, -1).sizes(), input.options(), MemoryFormat::Contiguous);
    input_.transpose_(-2, -1);
    input_.copy_(input);
  }

  // ormqr_stub (apply_ormqr) performs calculations in-place and 'result' must be a copy of 'other'
  result.copy_(other);

  ormqr_stub(result.device().type(), input_, tau_, result, left, transpose);
}

Tensor& ormqr_out(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose, Tensor& result) {
  TORCH_CHECK(input.dim() >= 2, "torch.ormqr: input must have at least 2 dimensions.");
  TORCH_CHECK(other.dim() >= 2, "torch.ormqr: other must have at least 2 dimensions.");

  int64_t left_size_condition = left ? -2 : -1;
  TORCH_CHECK(
      other.size(left_size_condition) >= tau.size(-1),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be greater than or equal to tau.shape[-1]");

  TORCH_CHECK(
      other.size(left_size_condition) == input.size(-2),
      "torch.ormqr: other.shape[",
      left_size_condition,
      "] must be equal to input.shape[-2]");

  TORCH_CHECK(
      input.dim() - tau.dim() == 1,
      "torch.ormqr: ",
      "Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      tau.dim(),
      " and input.ndim is equal to ",
      input.dim());
  TORCH_CHECK(
      input.dim() == other.dim(),
      "torch.ormqr: ",
      "Expected other to have the same number of dimensions as input, but got other.ndim equal to ",
      other.dim(),
      " and input.ndim is equal to ",
      input.dim());

  if (input.dim() > 2) {
    auto expected_batch_shape = IntArrayRef(input.sizes().data(), input.dim() - 2); // input.shape[:-2]
    auto actual_batch_tau_shape = IntArrayRef(tau.sizes().data(), tau.dim() - 1); // tau.shape[:-1]
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);

    auto actual_batch_other_shape = IntArrayRef(other.sizes().data(), other.dim() - 2); // other.shape[:-2]
    TORCH_CHECK(
        actual_batch_other_shape.equals(expected_batch_shape),
        "torch.ormqr: Expected batch dimensions of other to be equal to input.shape[:-2], but got ",
        actual_batch_other_shape);
  }

  TORCH_CHECK(
      tau.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and tau to have the same dtype, but input has dtype", input.scalar_type(),
      " and tau has dtype ", tau.scalar_type());
  TORCH_CHECK(
      other.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and other to have the same dtype, but input has dtype", input.scalar_type(),
      " and other has dtype ", other.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == input.scalar_type(),
      "torch.ormqr: Expected input and result to have the same dtype, but input has dtype", input.scalar_type(),
      " and result has dtype ", result.scalar_type());

  checkSameDevice("torch.ormqr", tau, input, "tau");
  checkSameDevice("torch.ormqr", other, input, "other");
  checkSameDevice("torch.ormqr", result, input);

  bool result_equal_expected_shape = result.sizes().equals(other.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape); // or result does not have the expected shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    ormqr_out_helper(input, tau, other, result_tmp, left, transpose);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    // use result's storage directly
    ormqr_out_helper(input, tau, other, result, left, transpose);
  }

  return result;
}

Tensor ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  Tensor result = at::empty({0}, input.options());
  result = at::native::ormqr_out(input, tau, other, left, transpose, result);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(linalg_eigh_stub);

/*
  Computes eigenvalues and eigenvectors of the tensor 'input'.

  Args:
  * 'input' - input Tensor for eigendecomposition
  * 'values' - Tensor to store computed eigenvalues
  * 'vectors' - Tensor to store computed eigenvectors
  * 'infos' - Tensor to store LAPACK/MAGMA/cuSOLVER error codes
  * 'compute_eigenvectors' - controls whether eigenvectors should be computed
  * 'uplo_str' - controls the portion of input matrix to consider in computations, allowed values are "u", "U", "l", "L"
    "u", "U" - upper triangular portion of the input matrix is used in computations; "l", "L" - lower.
*/
std::tuple<Tensor&, Tensor&> linalg_eigh_out_info(
    const Tensor& input,
    Tensor& values,
    Tensor& vectors,
    Tensor& infos,
    bool compute_eigenvectors,
    const std::string& uplo_str) {
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == vectors.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == values.device());

  // eigenvalues are always real-valued
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  ScalarType real_dtype = toValueType(input.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.scalar_type() == real_dtype);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == vectors.scalar_type());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == input.device());

  // infos can have the shape equal to input.shape[:-2] or (batchCount(input), ), both would work with the current implementation.
  // infos.shape == input.shape[:-2] might be useful in the future for easier checking the error code for the specific matrix
  // in batched input when we would have a user-exposed way to get infos tensor.
  // 1-dimensional tensor of shape (batchCount(input), ) is currently used for the internal implementation everywhere.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.numel() == std::max<int64_t>(1, batchCount(input)));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.is_contiguous());

  // if 'vectors' has no elements we can modify it
  if (vectors.numel() == 0) {
    vectors.resize_(input.sizes(), MemoryFormat::Contiguous);
    vectors.transpose_(-2, -1);  // make 'vectors' to have Fortran contiguous memory layout
  }

  // if 'values' has no elements we can modify it
  auto values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  if (values.numel() == 0) {
    values.resize_(values_shape, MemoryFormat::Contiguous);
  }

  // 'vectors' must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.sizes().equals(input.sizes()));

  // 'values' must be contiguous
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.sizes().equals(values_shape));

  // linalg_eigh_stub performs calculations in-place and 'vectors' must be a copy of 'input'
  vectors.copy_(input);

  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  char uplo = std::toupper(uplo_str[0]);
  bool upper = (uplo == 'U');

  linalg_eigh_stub(input.device().type(), values, vectors, infos, upper, compute_eigenvectors);

  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor, Tensor> linalg_eigh(const Tensor& input, std::string uplo) {
  squareCheckInputs(input);
  checkUplo(uplo);
  ScalarType real_dtype = toValueType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(real_dtype));
  Tensor vectors = at::empty({0}, input.options());
  Tensor infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt));

  std::tie(values, vectors) = linalg_eigh_out_info(input, values, vectors, infos, true, uplo);

  if (input.dim() > 2) {
    batchCheckErrors(infos, "torch.linalg.eigh");
  } else {
    singleCheckErrors(infos.item().toInt(), "torch.linalg.eigh");
  }

  return std::tuple<Tensor, Tensor>(values, vectors);
}

// TODO: it's possible to make the _out variant to be a primal function and implement linalg_eigh on top of _out
// TODO: implement _out variant avoiding copy and using already allocated storage directly
std::tuple<Tensor&, Tensor&> linalg_eigh_out(const Tensor& input, std::string uplo, Tensor& eigvals, Tensor& eigvecs) {
  checkSameDevice("torch.linalg.eigh", eigvecs, input, "eigenvectors");
  checkSameDevice("torch.linalg.eigh", eigvals, input, "eigenvalues");
  checkLinalgCompatibleDtype("torch.linalg.eigh", eigvecs, input, "eigenvectors");

  // eigenvalues are always real-valued here
  ScalarType real_dtype = toValueType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.eigh", eigvals.scalar_type(), real_dtype, "eigenvalues");

  Tensor eigvals_tmp, eigvecs_tmp;
  std::tie(eigvals_tmp, eigvecs_tmp) = at::linalg_eigh(input, uplo);

  at::native::resize_output(eigvals, eigvals_tmp.sizes());
  eigvals.copy_(eigvals_tmp);
  at::native::resize_output(eigvecs, eigvecs_tmp.sizes());
  eigvecs.copy_(eigvecs_tmp);

  return std::tuple<Tensor&, Tensor&>(eigvals, eigvecs);
}

Tensor linalg_eigvalsh(const Tensor& input, std::string uplo) {
  // if input requires grad we must compute the eigenvectors to make this function differentiable
  // the eigenvectors are not exposed to the user
  if (at::GradMode::is_enabled() && input.requires_grad()) {
    Tensor values;
    std::tie(values, std::ignore) = at::linalg_eigh(input, uplo);
    return values;
  }

  squareCheckInputs(input);
  checkUplo(uplo);
  ScalarType real_dtype = toValueType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(real_dtype));
  Tensor vectors = at::empty({0}, input.options());
  Tensor infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt));

  std::tie(values, vectors) = linalg_eigh_out_info(input, values, vectors, infos, false, uplo);

  if (input.dim() > 2) {
    batchCheckErrors(infos, "torch.linalg.eigvalsh");
  } else {
    singleCheckErrors(infos.item().toInt(), "torch.linalg.eigvalsh");
  }

  return values;
}

// TODO: it's possible to make the _out variant to be a primal function and implement linalg_eigvalsh on top of _out
// TODO: implement _out variant avoiding copy and using already allocated storage directly
Tensor& linalg_eigvalsh_out(const Tensor& input, std::string uplo, Tensor& result) {
  checkSameDevice("torch.linalg.eigvalsh", result, input);
  ScalarType real_dtype = toValueType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.eigvalsh", result.scalar_type(), real_dtype);

  Tensor result_tmp = at::linalg_eigvalsh(input, uplo);

  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ symeig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_symeig(Tensor& self, Tensor& eigvals, bool eigenvectors, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("symeig: LAPACK library not found in compilation");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto eigvals_data = eigvals.data_ptr<value_t>();
  auto self_matrix_stride = matrixStride(self);
  auto eigvals_stride = eigvals.size(-1);
  auto batch_size = batchCount(self);
  auto n = self.size(-1);

  char uplo = upper ? 'U' : 'L';
  char jobz = eigenvectors ? 'V' : 'N';

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (isComplexType(at::typeMetaToScalarType(self.dtype()))) {
    int64_t lrwork = std::max(int64_t(1), 3 * n - 2);
    ScalarType dtype = toValueType(typeMetaToScalarType(self.dtype()));
    rwork = at::empty({lrwork}, self.options().dtype(dtype));
    rwork_data = rwork.data_ptr<value_t>();
  }

  lapackSymeig<scalar_t, value_t>(jobz, uplo, n, self_data, n, eigvals_data, &wkopt, lwork, rwork_data, &info);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    value_t* eigvals_working_ptr = &eigvals_data[i * eigvals_stride];

    // now compute the eigenvalues and the eigenvectors (optionally)
    lapackSymeig<scalar_t, value_t>(jobz, uplo, n, self_working_ptr, n, eigvals_working_ptr, work.data_ptr<scalar_t>(), lwork, rwork_data, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

std::tuple<Tensor, Tensor> _symeig_helper_cpu(const Tensor& self, bool eigenvectors, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);

  auto self_sizes = self.sizes().vec();
  self_sizes.pop_back();
  ScalarType dtype = toValueType(typeMetaToScalarType(self.dtype()));
  auto eigvals = at::empty(self_sizes, self.options().dtype(dtype));

  if (self.numel() == 0) {
    return std::tuple<Tensor, Tensor>(eigvals, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  }

  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "symeig_cpu", [&]{
    apply_symeig<scalar_t>(self_working_copy, eigvals, eigenvectors, upper, infos);
  });

  if (self.dim() > 2) {
    batchCheckErrors(infos, "symeig_cpu");
  } else {
    singleCheckErrors(infos[0], "symeig_cpu");
  }
  if (eigenvectors) {
    return std::tuple<Tensor, Tensor>(eigvals, self_working_copy);
  } else {
    return std::tuple<Tensor, Tensor>(eigvals, at::empty({0}, self.options()));
  }
}

std::tuple<Tensor, Tensor> symeig(const Tensor& self, bool eigenvectors, bool upper) {
  TORCH_WARN_ONCE(
    "torch.symeig is deprecated in favor of torch.linalg.eigh and will be removed in a future ",
    "PyTorch release.\n",
    "The default behavior has changed from using the upper triangular portion of the matrix by default ",
    "to using the lower triangular portion.\n",
    "L, _ = torch.symeig(A, upper=upper)\n",
    "should be replaced with\n",
    "L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n",
    "and\n",
    "L, V = torch.symeig(A, eigenvectors=True)\n"
    "should be replaced with\n",
    "L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')"
  );
  squareCheckInputs(self);
  return at::_symeig_helper(self, eigenvectors, upper);
}

std::tuple<Tensor&, Tensor&> symeig_out(const Tensor& self, bool eigenvectors, bool upper, Tensor& vals, Tensor& vecs) {
  TORCH_WARN_ONCE(
    "torch.symeig is deprecated in favor of torch.linalg.eigh and will be removed in a future ",
    "PyTorch release.\n",
    "The default behavior has changed from using the upper triangular portion of the matrix by default ",
    "to using the lower triangular portion.\n",
    "L, _ = torch.symeig(A, upper=upper)\n",
    "should be replaced with\n",
    "L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')\n",
    "and\n",
    "L, V = torch.symeig(A, eigenvectors=True)\n"
    "should be replaced with\n",
    "L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')"
  );
  checkSameDevice("symeig", vals, self, "eigenvalues");
  checkSameDevice("symeig", vecs, self, "eigenvectors");
  checkLinalgCompatibleDtype("symeig", vecs, self, "eigenvectors");
  // eigenvalues are always real-valued here
  ScalarType real_dtype = toValueType(self.scalar_type());
  checkLinalgCompatibleDtype("symeig", vals.scalar_type(), real_dtype, "eigenvalues");

  Tensor vals_tmp, vecs_tmp;
  std::tie(vals_tmp, vecs_tmp) = at::symeig(self, eigenvectors, upper);

  at::native::resize_output(vals, vals_tmp.sizes());
  at::native::resize_output(vecs, vecs_tmp.sizes());
  vals.copy_(vals_tmp);
  vecs.copy_(vecs_tmp);
  return std::tuple<Tensor&, Tensor&>(vals, vecs);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This function returns complex-valued eigenvectors that is obtained from LAPACK GEEV's real-valued output
// This function is also used for the MAGMA path because intermediate MAGMA's results live on CPU
template <typename scalar_t>
static void linalg_eig_make_complex_eigenvectors_impl(Tensor& result, const Tensor& complex_values, const Tensor& real_vectors) {
  // From GEEV documentation:
  // Complex conjugate pairs of eigenvalues appear consecutively with the eigenvalue having the positive imaginary part first
  // If the j-th eigenvalue is real, then v(j) = VR(:,j), the j-th column of VR.
  // If the j-th and (j+1)-st eigenvalues form a complex conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and v(j+1) = VR(:,j) - i*VR(:,j+1).

  auto batch_size = batchCount(real_vectors);
  auto n = real_vectors.size(-1);
  auto matrix_stride = matrixStride(real_vectors);

  auto result_data = result.data_ptr<c10::complex<scalar_t>>();
  auto real_vectors_data = real_vectors.data_ptr<scalar_t>();
  auto values_data = complex_values.data_ptr<c10::complex<scalar_t>>();

  for (auto b = decltype(batch_size){0}; b < batch_size; b++) {
    scalar_t* vecs = &real_vectors_data[b * matrix_stride];
    c10::complex<scalar_t>* res = &result_data[b * matrix_stride];
    c10::complex<scalar_t>* vals = &values_data[b * n];
    for (auto j = decltype(n){0}; j < n; j++) {
      if (vals[j].imag() == 0.0) {  // eigenvalue is real, then v(j) = VR(:,j)
        for (auto i = decltype(n){0}; i < n; i++) {
          res[j * n + i] = c10::complex<scalar_t>(vecs[j * n + i], 0);
        }
      } else {
        for (auto i = decltype(n){0}; i < n; i++) {
          res[j * n + i] = c10::complex<scalar_t>(vecs[j * n + i],  vecs[(j+1) * n + i]);      // v(j)   = VR(:,j) + i*VR(:,j+1)
          res[(j+1) * n + i] = c10::complex<scalar_t>(vecs[j * n + i], -vecs[(j+1) * n + i]);  // v(j+1) = VR(:,j) - i*VR(:,j+1)
        }
        j++;
      }
    }
  }
}

static Tensor& linalg_eig_make_complex_eigenvectors(Tensor& complex_vectors, const Tensor& complex_values, const Tensor& real_vectors) {
  // These asserts make explicit the requirements on tensors for 'linalg_eig_make_complex_eigenvectors_impl'
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.device() == at::kCPU);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.is_complex());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_complex());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.is_floating_point());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.transpose(-2, -1).is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.transpose(-2, -1).is_contiguous());

  AT_DISPATCH_FLOATING_TYPES(real_vectors.scalar_type(), "linalg_eig_make_complex_vector", [&]{
    linalg_eig_make_complex_eigenvectors_impl<scalar_t>(complex_vectors, complex_values, real_vectors);
  });
  return complex_vectors;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(linalg_eig_stub);

std::tuple<Tensor&, Tensor&> linalg_eig_out_info(const Tensor& input, Tensor& values, Tensor& vectors, Tensor& infos, bool compute_eigenvectors) {
  // MAGMA doesn't have GPU interface for GEEV routine, it requires inputs to be on CPU
  // therefore we create all intermediate tensors on CPU
  auto options = input.options().device(at::kCPU);

  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of the hierarchy of calls
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));

  // for real-valued 'input', eigenvalues can be real-valued or complex-valued
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((toComplexType(input.scalar_type()) == values.scalar_type()) || (input.scalar_type() == values.scalar_type()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == at::kCPU);

  // for real-valued 'input', eigenvectors can be real-valued or complex-valued
  if (compute_eigenvectors) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY((toComplexType(input.scalar_type()) == vectors.scalar_type()) || (input.scalar_type() == vectors.scalar_type()));
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.device() == at::kCPU);
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.numel() == std::max<int64_t>(1, batchCount(input)));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.is_contiguous());

  // if 'vectors' has no elements we can modify it
  if (vectors.numel() == 0 && compute_eigenvectors) {
    vectors.resize_(input.sizes(), MemoryFormat::Contiguous);
    vectors.transpose_(-2, -1);  // make 'vectors' to have Fortran contiguous memory layout
  }

  // if 'values' has no elements we can modify it
  auto values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  if (values.numel() == 0) {
    values.resize_(values_shape, MemoryFormat::Contiguous);
  }

  // 'vectors' must be in batched column major order (Fortran contiguous)
  if (compute_eigenvectors) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.transpose(-2, -1).is_contiguous());
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.sizes().equals(input.sizes()));
  }

  // 'values' must be contiguous
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.sizes().equals(values_shape));

  // if 'input' is complex then use 'values' directly else create a temporary to hold the real and imaginary parts
  // and then use at::complex_out
  Tensor real_imag_values = values;

  // if 'input' is complex then use 'vectors' directly else maybe create a temporary to hold real vectors
  // and then use linalg_eig_make_complex_eigenvectors
  Tensor maybe_complex_vectors = vectors;
  if (!input.is_complex()) {
    // first n elements to hold the real portion of the output and the last n elements to hold the imaginary portion
    auto real_imag_shape = IntArrayRef(input.sizes().data(), input.dim()-2).vec();  // input.shape[:-2]
    real_imag_shape.push_back(input.size(-1) * 2);
    real_imag_values = at::empty(real_imag_shape, options, MemoryFormat::Contiguous);

    // linalg_eig_stub expects real-valued tensor to store eigenvectors
    // output of linalg_eig_stub need to be post-processed later to produce complex-valued eigenvectors
    // we do this post-processing only if 'vectors' is complex-valued
    // otherwise storage of 'vectors' is used directly
    if (vectors.is_complex() && compute_eigenvectors) {
      maybe_complex_vectors = at::empty(input.sizes(), options, MemoryFormat::Contiguous);
      maybe_complex_vectors.transpose_(-2, -1);  // make 'maybe_complex_vectors' to have Fortran contiguous memory layout
    }
  }

  // MAGMA uses a hybrid CPU-GPU algorithm that performs well only for large matrices
  // See: https://github.com/pytorch/pytorch/pull/52491#issuecomment-795685687
  // Here we call CPU path for matrices smaller than 2048x2048
  // that should be in general significantly faster than calling MAGMA
  if (input.size(-1) <= 2048) {
    linalg_eig_stub(at::kCPU, real_imag_values, maybe_complex_vectors, infos, input.to(kCPU), compute_eigenvectors);
  } else {
    linalg_eig_stub(input.device().type(), real_imag_values, maybe_complex_vectors, infos, input, compute_eigenvectors);
  }

  // if input is not complex we need to do some post-processing
  if (!input.is_complex()) {
    // extract real and imaginary parts of the output
    auto real_values = real_imag_values.slice(/*dim=*/-1, /*start=*/0, /*end*/input.size(-1));
    auto imag_values = real_imag_values.slice(/*dim=*/-1, /*start=*/input.size(-1));

    // if the imaginary part is zero we don't need to do anything
    bool is_zero_imag = at::all(imag_values == 0.0).item().toBool();
    if (is_zero_imag) {
      values.copy_(real_values);
      if (compute_eigenvectors) {
        vectors.copy_(maybe_complex_vectors);  // does nothing for !vectors.is_complex() because vectors.is_same(maybe_complex_vectors) == true
      }
      return std::tuple<Tensor&, Tensor&>(values, vectors);
    }

    if (values.is_complex()) {
      values = at::complex_out(values, real_values, imag_values);
    } else {
      TORCH_CHECK(false, "torch.linalg.eig: imaginary part of eigenvalues is non-zero, can't safely cast eigenvalues to non-complex dtype.")
    }
    if (compute_eigenvectors) {
      if (vectors.is_complex()) {
          vectors = linalg_eig_make_complex_eigenvectors(vectors, values, maybe_complex_vectors);
      } else {
        TORCH_CHECK(false, "torch.linalg.eig: imaginary part of eigenvectors is non-zero, can't safely cast eigenvectors to non-complex dtype.")
      }
    }
  }

  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor&, Tensor&> linalg_eig_out(const Tensor& input, Tensor& values, Tensor& vectors) {
  squareCheckInputs(input);

  // unlike NumPy for real-valued inputs the output is always complex-valued
  checkLinalgCompatibleDtype("torch.linalg.eig", values.scalar_type(), toComplexType(input.scalar_type()), "eigenvalues");
  checkLinalgCompatibleDtype("torch.linalg.eig", vectors.scalar_type(), toComplexType(input.scalar_type()), "eigenvectors");
  checkSameDevice("torch.linalg.eig", values, input, "eigenvalues");
  checkSameDevice("torch.linalg.eig", vectors, input, "eigenvectors");

  // MAGMA doesn't have GPU interface for GEEV routine, it requires inputs to be on CPU
  auto options = input.options().device(at::kCPU);
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, options.dtype(kInt));

  // if result is not empty and not in batched column major format we have to allocate a temporary tensor
  bool is_batched_column_major = false;
  if (vectors.dim() >= 2) {
    is_batched_column_major = vectors.transpose(-2, -1).is_contiguous();
  }

  bool values_expected_type = (values.scalar_type() == toComplexType(input.scalar_type()));
  bool vectors_expected_type = (vectors.scalar_type() == toComplexType(input.scalar_type()));

  auto expected_values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  bool values_equal_expected_shape = values.sizes().equals(expected_values_shape);
  bool vectors_equal_expected_shape = vectors.sizes().equals(input.sizes());

  // if result is not empty and not in batched column major format
  bool values_tmp_needed = (values.numel() != 0 && !values.is_contiguous());
  bool vectors_tmp_needed = (vectors.numel() != 0 && !is_batched_column_major);
  // or result does not have the expected shape
  values_tmp_needed |= (values.numel() != 0 && !values_equal_expected_shape);
  vectors_tmp_needed |= (vectors.numel() != 0 && !vectors_equal_expected_shape);
  // or result does not have the expected dtype
  values_tmp_needed |= !values_expected_type;
  vectors_tmp_needed |= !vectors_expected_type;
  // we will allocate a temporary tensor and do the copy

  // because MAGMA's GEEV takes CPU inputs and returns CPU outputs
  // "out" tensors that are on GPU device can't be used directly
  values_tmp_needed |= values.is_cuda();
  vectors_tmp_needed |= vectors.is_cuda();

  // determine the appropriate scalar_type for the temporary tensors
  ScalarType values_type = input.scalar_type();
  ScalarType vectors_type = input.scalar_type();
  if (!input.is_complex()) {
    // for real-valued input we can have either real- or complex-valued output
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
    vectors_type = vectors.is_complex() ? input_complex_dtype : vectors_type;
  }

  if (values_tmp_needed && vectors_tmp_needed) {
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    Tensor vectors_tmp = at::empty({0}, options.dtype(vectors_type));
    std::tie(values_tmp, vectors_tmp) = linalg_eig_out_info(input, values_tmp, vectors_tmp, infos, true);
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
    at::native::resize_output(vectors, vectors_tmp.sizes());
    vectors.copy_(vectors_tmp);
  } else if (!values_tmp_needed && vectors_tmp_needed) {
    // use 'values' storage directly
    Tensor vectors_tmp = at::empty({0}, options.dtype(vectors_type));
    std::tie(values, vectors_tmp) = linalg_eig_out_info(input, values, vectors_tmp, infos, true);
    at::native::resize_output(vectors, vectors_tmp.sizes());
    vectors.copy_(vectors_tmp);
  } else if (values_tmp_needed && !vectors_tmp_needed) {
    // use 'vectors' storage directly
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    std::tie(values_tmp, vectors) = linalg_eig_out_info(input, values_tmp, vectors, infos, true);
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
  } else {
    // use 'values' and 'vectors' storage directly
    std::tie(values, vectors) = linalg_eig_out_info(input, values, vectors, infos, true);
  }

  // Now check LAPACK/MAGMA error codes
  if (input.dim() > 2) {
    batchCheckErrors(infos, "torch.linalg.eig");
  } else {
    singleCheckErrors(infos.item().toInt(), "torch.linalg.eig");
  }

  return std::tuple<Tensor&, Tensor&>(values, vectors);
}

std::tuple<Tensor, Tensor> linalg_eig(const Tensor& input) {
  ScalarType complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));
  Tensor vectors = at::empty({0}, input.options().dtype(complex_dtype));

  at::linalg_eig_outf(input, values, vectors);

  return std::tuple<Tensor, Tensor>(values, vectors);
}

Tensor& linalg_eigvals_out(const Tensor& input, Tensor& values) {
  squareCheckInputs(input);

  // unlike NumPy for real-valued inputs the output is always complex-valued
  checkLinalgCompatibleDtype("torch.linalg.eigvals", values.scalar_type(), toComplexType(input.scalar_type()), "eigenvalues");
  checkSameDevice("torch.linalg.eigvals", values, input, "eigenvalues");

  // MAGMA doesn't have GPU interface for GEEV routine, it requires inputs to be on CPU
  auto options = input.options().device(at::kCPU);
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, options.dtype(kInt));

  bool values_expected_type = (values.scalar_type() == toComplexType(input.scalar_type()));

  auto expected_values_shape = IntArrayRef(input.sizes().data(), input.dim()-1);  // input.shape[:-1]
  bool values_equal_expected_shape = values.sizes().equals(expected_values_shape);

  // if result is not empty and not in batched column major format
  bool values_tmp_needed = (values.numel() != 0 && !values.is_contiguous());
  // or result does not have the expected shape
  values_tmp_needed |= (values.numel() != 0 && !values_equal_expected_shape);
  // or result does not have the expected dtype
  values_tmp_needed |= !values_expected_type;
  // we will allocate a temporary tensor and do the copy

  // because MAGMA's GEEV takes CPU inputs and returns CPU outputs
  // 'values' tensor that is on GPU device can't be used directly
  values_tmp_needed |= values.is_cuda();

  // determine the appropriate scalar_type for the temporary tensors
  ScalarType values_type = input.scalar_type();
  if (!input.is_complex()) {
    // for real-valued input we can have either real- or complex-valued output
    ScalarType input_complex_dtype = toComplexType(input.scalar_type());
    values_type = values.is_complex() ? input_complex_dtype : values_type;
  }

  Tensor vectors;
  if (values_tmp_needed) {
    Tensor values_tmp = at::empty({0}, options.dtype(values_type));
    std::tie(values_tmp, std::ignore) = linalg_eig_out_info(input, values_tmp, vectors, infos, /*compute_eigenvectors=*/false);
    at::native::resize_output(values, values_tmp.sizes());
    values.copy_(values_tmp);
  } else { // use 'values' storage directly
    std::tie(values, std::ignore) = linalg_eig_out_info(input, values, vectors, infos, /*compute_eigenvectors=*/false);
  }

  // Now check LAPACK/MAGMA error codes
  if (input.dim() > 2) {
    batchCheckErrors(infos, "torch.linalg.eigvals");
  } else {
    singleCheckErrors(infos.item().toInt(), "torch.linalg.eigvals");
  }

  return values;
}

Tensor linalg_eigvals(const Tensor& input) {
  // if input requires grad we must compute the eigenvectors to make this function differentiable
  // the eigenvectors are not exposed to the user
  if (at::GradMode::is_enabled() && input.requires_grad()) {
    return std::get<0>(at::linalg_eig(input));
  }

  ScalarType complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));

  at::linalg_eigvals_outf(input, values);

  return values;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(eig_stub);

std::tuple<Tensor&, Tensor&> eig_out(const Tensor& self, bool eigenvectors, Tensor& e, Tensor& v) {
  TORCH_WARN_ONCE(
    "torch.eig is deprecated in favor of torch.linalg.eig and will be removed in a future ",
    "PyTorch release.\n",
    "torch.linalg.eig returns complex tensors of dtype cfloat or cdouble rather than real tensors ",
    "mimicking complex tensors.\n",
    "L, _ = torch.eig(A)\n",
    "should be replaced with\n",
    "L_complex = torch.linalg.eigvals(A)\n",
    "and\n",
    "L, V = torch.eig(A, eigenvectors=True)\n",
    "should be replaced with\n",
    "L_complex, V_complex = torch.linalg.eig(A)"
  );
  TORCH_CHECK(self.dim() == 2, "input should be 2 dimensional");
  TORCH_CHECK(self.size(0) == self.size(1), "input should be square");
  TORCH_CHECK(self.isfinite().all().item<bool>(), "input should not contain infs or NaNs");
  checkSameDevice("torch.eig", e, self, "eigenvalues");
  checkLinalgCompatibleDtype("torch.eig", e, self, "eigenvalues");
  if (eigenvectors) {
    checkSameDevice("torch.eig", v, self, "eigenvectors");
    checkLinalgCompatibleDtype("torch.eig", v, self, "eigenvectors");
  }
  int64_t n = self.size(-1);

  if (isComplexType(at::typeMetaToScalarType(self.dtype()))) {
      at::native::resize_output(e, {n});
  } else {
      at::native::resize_output(e, {n, 2});
  }
  if (eigenvectors) {
      at::native::resize_output(v, self.sizes());
  }

  // optimization: if self is empty, we can immediately return the empty
  // tensors, instead of getting empty tensors from eig_helper
  if (self.numel() == 0) {
      return std::tuple<Tensor&, Tensor&>(e, v);
  }

  Tensor vals_, vecs_;
  std::tie(vals_, vecs_) = eig_stub(self.device().type(), self, eigenvectors);
  e.copy_(vals_);
  if (eigenvectors) {
    v.copy_(vecs_);
  }
  return std::tuple<Tensor&, Tensor&>(e, v);
}

std::tuple<Tensor,Tensor> eig(const Tensor& self, bool eigenvectors) {
  Tensor e = at::empty({0}, self.options());
  Tensor v = at::empty({0}, self.options());
  at::eig_out(e, v, self, eigenvectors);
  return std::tuple<Tensor, Tensor>(e, v);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_svd(Tensor& self, Tensor& U, Tensor& S, Tensor& VT,
                      char jobz, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("svd: LAPACK library not found in compilation");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);
  auto batchsize = batchCount(self);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto lda = std::max<int64_t>(1, m);
  auto ldvt = std::max<int64_t>(1, n);
  auto mn = std::min(m, n);
  Tensor iwork = at::empty({8 * mn}, at::kInt);
  auto iwork_data = iwork.data_ptr<int>();
  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (isComplexType(at::typeMetaToScalarType(self.dtype()))) {
    auto lrwork  = computeLRWorkDim(jobz, m, n);
    // rwork is an array of floats or doubles depending on the type
    rwork = at::empty({std::max(int64_t(1), lrwork)}, at::typeMetaToScalarType(S.dtype()));
    rwork_data = rwork.data_ptr<value_t>();
  }

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackSvd<scalar_t, value_t>(jobz, m, n, self_data, lda, S_data, U_data, lda, VT_data, ldvt, &wkopt, lwork, rwork_data, iwork_data, &info);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());
  auto work_data = work.data_ptr<scalar_t>();

  for (const auto i : c10::irange(batchsize)) {
    scalar_t* self_working_ptr = &self_data[i * self_stride];
    value_t* S_working_ptr = &S_data[i * S_stride];
    scalar_t* U_working_ptr = &U_data[i * U_stride];
    scalar_t* VT_working_ptr = &VT_data[i * VT_stride];

    // Compute S, U (optionally) and VT (optionally)
    lapackSvd<scalar_t, value_t>(jobz, m, n, self_working_ptr, lda,
                        S_working_ptr, U_working_ptr, lda, VT_working_ptr, ldvt, work_data, lwork, rwork_data, iwork_data, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper_cpu(const Tensor& self, bool some, bool compute_uv) {
  std::vector<int64_t> infos(batchCount(self), 0);
  int64_t m = self.size(-2), n = self.size(-1);
  int64_t k = std::min(m, n);

  char jobz = compute_uv ? (some ? 'S' : 'A') : 'N';

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) = _create_U_S_VT(self, some, compute_uv);

  auto self_working_copy = cloneBatchedColumnMajor(self);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cpu", [&]{
    apply_svd<scalar_t>(self_working_copy, U_working_copy, S_working_copy, VT_working_copy, jobz, infos);
  });

  if (self.dim() > 2) {
    batchCheckErrors(infos, "svd_cpu");
  } else {
    singleCheckErrors(infos[0], "svd_cpu");
  }

  if (!compute_uv) {
    VT_working_copy.zero_();
    U_working_copy.zero_();
  }

  if (some) {
    VT_working_copy = VT_working_copy.narrow(-2, 0, k);
  }

  // so far we have computed VT, but torch.svd returns V instead. Adjust accordingly.
  // Note that the 'apply_svd' routine returns VT = V^T (for real inputs) or VT = V^H (for complex inputs), not V.
  VT_working_copy = VT_working_copy.conj();
  VT_working_copy.transpose_(-2, -1);
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& self, bool some, bool compute_uv) {
  // TODO: uncomment the following when svd is deprecated not only in docs
  // torch/xla is blocking the transition from at::svd to at::linalg_svd in at::linalg_pinv code
  // see https://github.com/pytorch/xla/issues/2755
  // TORCH_WARN_ONCE(
  //     "torch.svd is deprecated in favor of torch.linalg.svd and will be ",
  //     "removed in a future PyTorch release.\n",
  //     "U, S, V = torch.svd(A, some=some, compute_uv=True) (default)\n",
  //     "should be replaced with\n",
  //     "U, S, Vh = torch.linalg.svd(A, full_matrices=not some)\n",
  //     "V = Vh.transpose(-2, -1).conj()\n",
  //     "and\n",
  //     "_, S, _ = torch.svd(A, some=some, compute_uv=False)\n",
  //     "should be replaced with\n",
  //     "S = torch.linalg.svdvals(A)");

  TORCH_CHECK(self.dim() >= 2,
              "svd input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  return at::_svd_helper(self, some, compute_uv);
}

std::tuple<Tensor&, Tensor&, Tensor&> svd_out(const Tensor& self, bool some, bool compute_uv, Tensor& U, Tensor& S, Tensor& V) {
  checkSameDevice("svd", U, self, "U");
  checkSameDevice("svd", S, self, "S");
  checkSameDevice("svd", V, self, "V");
  checkLinalgCompatibleDtype("svd", U, self, "U");
  checkLinalgCompatibleDtype("svd", V, self, "V");
  // singular values are always real-valued here
  ScalarType real_dtype = toValueType(self.scalar_type());
  checkLinalgCompatibleDtype("svd", S.scalar_type(), real_dtype, "S");

  Tensor U_tmp, S_tmp, V_tmp;
  std::tie(U_tmp, S_tmp, V_tmp) = at::native::svd(self, some, compute_uv);

  at::native::resize_output(U, U_tmp.sizes());
  at::native::resize_output(S, S_tmp.sizes());
  at::native::resize_output(V, V_tmp.sizes());
  U.copy_(U_tmp);
  S.copy_(S_tmp);
  V.copy_(V_tmp);
  return std::tuple<Tensor&, Tensor&, Tensor&>(U, S, V);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/* torch.linalg.svd, implemented in terms of torch.svd. There are two main
   differences:

    1. the 2nd parameter is bool some=True, which if effectively the opposite
       of full_matrices=True

    2. svd returns V, while linalg.svd returns Vh = V^T (for real inputs) or Vh = V^H (for complex inputs).
       To accommodate the difference, we transpose() and conj() V upon return
*/

std::tuple<Tensor, Tensor, Tensor> linalg_svd(const Tensor& self, bool full_matrices) {
  TORCH_CHECK(self.dim() >= 2,
              "svd input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");

    bool some = !full_matrices;
    Tensor U, S, V;
    std::tie(U, S, V) = at::_svd_helper(self, some, /*compute_uv=*/true);

    Tensor Vh = V.conj().transpose(-2, -1);
    return std::make_tuple(U, S, Vh);

}

static void svd_resize_and_copy(const char *name, const Tensor& src, Tensor &dst) {
  TORCH_CHECK(src.device() == dst.device(), "svd output tensor ", name, " is on the wrong device: expected ", src.device(), " got ", dst.device());
  at::native::resize_output(dst, src.sizes());
  dst.copy_(src);
}

std::tuple<Tensor&, Tensor&, Tensor&> linalg_svd_out(const Tensor& self, bool full_matrices, Tensor& U, Tensor& S, Tensor& Vh) {
  checkSameDevice("svd", U, self, "U");
  checkSameDevice("svd", S, self, "S");
  checkSameDevice("svd", Vh, self, "Vh");
  checkLinalgCompatibleDtype("linalg_svd", U, self, "U");
  checkLinalgCompatibleDtype("linalg_svd", Vh, self, "Vh");
  // singular values are always real-valued here
  ScalarType real_dtype = toValueType(self.scalar_type());
  checkLinalgCompatibleDtype("linalg_svd", S.scalar_type(), real_dtype, "S");
  Tensor U_tmp, S_tmp, Vh_tmp;
  std::tie(U_tmp, S_tmp, Vh_tmp) = at::native::linalg_svd(self, full_matrices);
  svd_resize_and_copy("U", U_tmp, U);
  svd_resize_and_copy("S", S_tmp, S);
  svd_resize_and_copy("V", Vh_tmp, Vh);
  return std::tuple<Tensor&, Tensor&, Tensor&>(U, S, Vh);
}

Tensor linalg_svdvals(const Tensor& input) {
  TORCH_CHECK(
      input.dim() >= 2,
      "torch.linalg.svdvals: input should have at least 2 dimensions, but has ",
      input.dim(),
      " dimensions instead");

  Tensor singular_values;

  // if input requires grad we must compute the singular vectors to make this function differentiable
  // the singular vectors are not exposed to the user
  const bool input_requires_grad = (at::GradMode::is_enabled() && input.requires_grad());
  std::tie(std::ignore, singular_values, std::ignore) =
      at::_svd_helper(input, /*some=*/input_requires_grad, /*compute_uv=*/input_requires_grad);
  return singular_values;
}

Tensor& linalg_svdvals_out(const Tensor& input, Tensor& result) {
  checkSameDevice("torch.linalg.svdvals", result, input);

  // singular values are always real-valued
  ScalarType real_dtype = toValueType(input.scalar_type());
  checkLinalgCompatibleDtype(
      "torch.linalg.svdvals", result.scalar_type(), real_dtype);

  Tensor singular_values_tmp;
  std::tie(std::ignore, singular_values_tmp, std::ignore) =
      // NOLINTNEXTLINE(bugprone-argument-comment)
      at::_svd_helper(input, /*full_matrices=*/false, /*compute_uv=*/false);

  at::native::resize_output(result, singular_values_tmp.sizes());
  result.copy_(singular_values_tmp);

  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lstsq_stub);

/*
  Solves a least squares problem. That is minimizing the squared Frobenius norm of |B - A X|.

  Input args:
  * 'input' - Tensor containing batches of m-by-n matrix A.
  * 'other' - Tensor containing batches of max(m, n)-by-nrhs matrix B.
  * 'cond' - relative tolerance for determining rank of A.
  * 'driver' - the name of the LAPACK driver that is used to compute the solution.
  Output args (modified in-place):
  * 'solution' - Tensor to store the solution matrix X.
  * 'residuals' - Tensor to store values of the residual sum of squares for each column of the solution.
  * 'rank' - Tensor to store the rank of A.
  * 'singular_values' - Tensor to store the singular values of A.
  * 'infos' - Tensor to store error codes of linear algebra math library.

  For further details, please see the LAPACK documentation for GELS/GELSY/GELSS/GELSD routines.
*/
static void linalg_lstsq_out_info(
    Tensor& solution,
    Tensor& residuals,
    Tensor& rank,
    Tensor& singular_values,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other,
    double rcond,
    std::string& driver) {
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(other.dim() >= 1);

  auto dim_diff = input.dim() - other.dim();
  TORCH_INTERNAL_ASSERT(0 <= dim_diff && dim_diff <= 1);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == other.device());

  TORCH_INTERNAL_ASSERT(solution.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(solution.device() == input.device());

  TORCH_INTERNAL_ASSERT(residuals.device() == input.device());

  TORCH_INTERNAL_ASSERT(rank.scalar_type() == at::kLong);
  TORCH_INTERNAL_ASSERT(rank.device() == input.device());

  auto real_dtype = toValueType(input.scalar_type());
  TORCH_INTERNAL_ASSERT(singular_values.scalar_type() == real_dtype);
  TORCH_INTERNAL_ASSERT(singular_values.device() == input.device());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == input.device());
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());

  bool vector_case = linalg_solve_is_vector_rhs(input, other);
  // we need to unsqueeze 'other' because 2-dimensional tensors are expected in the implementation
  Tensor other_2d = vector_case ? other.unsqueeze(-1) : other;

  TORCH_INTERNAL_ASSERT(input.size(-2) == other_2d.size(-2));

  std::vector<int64_t> expected_solution_shape = broadcast_batch_size(input, other_2d, input.dim() - 2);
  // the actual shape of the solution returned is (*, n,) or (*, n, nrhs)
  // but LAPACK requires extra dimensions to store raw residuals
  // so the expected shape is (*, max(m, n),) or (*, max(m, n), nrhs)
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto nrhs = other.size(-1);
  expected_solution_shape.push_back(std::max(m, n));
  if (!vector_case) {
    expected_solution_shape.push_back(nrhs);
  }

  // if 'solution' has no elements we can modify it
  if (solution.numel() == 0) {
    if (vector_case) {
      solution.resize_(expected_solution_shape, MemoryFormat::Contiguous);
    } else {
      auto shape_transposed = expected_solution_shape;
      std::swap(shape_transposed.end()[-1], shape_transposed.end()[-2]);
      solution.resize_(shape_transposed, MemoryFormat::Contiguous);
      solution.transpose_(-2, -1);
    }
  }

  // if 'solution' is non-empty it must have the expected shape
  TORCH_INTERNAL_ASSERT(solution.sizes().equals(expected_solution_shape));

  // 'solution' must be in batched column major order (Fortran contiguous) for 2D inputs
  // or C contiguous for 1D input
  if (vector_case) {
    TORCH_INTERNAL_ASSERT(solution.is_contiguous());
  } else {
    TORCH_INTERNAL_ASSERT(solution.transpose(-2, -1).is_contiguous());
  }

  // for 1-dimensional 'other', we need to unsqueeze the 'solution' before passing to "apply_solve"
  if (vector_case) {
    solution = solution.unsqueeze_(-1);
  }

  // _linalg_lstsq_helper_ performs calculations in-place and 'solution' must be a copy of other_2d
  solution.narrow(-2, 0, other_2d.size(-2)).copy_(other_2d);

  // if 'rank' is empty we might resize it
  auto input_batch_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);
  if (rank.numel() == 0 && driver != "gels") { // gels driver doesn't set 'rank'
    rank.resize_(input_batch_shape, MemoryFormat::Contiguous);
  }

  // if 'rank' is non-empty it must have the expected shape and be contiguous
  if (driver != "gels") {
    TORCH_INTERNAL_ASSERT(rank.sizes().equals(input_batch_shape));
    TORCH_INTERNAL_ASSERT(rank.is_contiguous());
  }

  // if 'singular_values' is empty we might resize it
  auto singular_values_shape = input_batch_shape.vec();
  singular_values_shape.push_back(std::min(m, n));
  if (singular_values.numel() == 0 && (driver == "gelsd" || driver == "gelss")) {
    singular_values.resize_(singular_values_shape, MemoryFormat::Contiguous);
  }

  // if 'singular_values' is non-empty it must have the expected shape and be contiguous
  if (driver == "gelsd" || driver == "gelss") {
    TORCH_INTERNAL_ASSERT(singular_values.sizes().equals(singular_values_shape));
    TORCH_INTERNAL_ASSERT(singular_values.is_contiguous());
  }

  // 'input' is modified in-place so we need a column-major copy
  auto input_working_copy = copyBatchedColumnMajor(input);

  // now the actual call that computes the result in-place (apply_lstsq)
  lstsq_stub(input.device().type(), input_working_copy, solution, rank, singular_values, infos, rcond, driver);

  // residuals are available only if m > n and drivers other than gelsy used
  if (m > n && driver != "gelsy") {
    // if the driver is gelss or gelsd then the residuals are available only if rank == n
    bool compute_residuals = true;
    if (driver == "gelss" || driver == "gelsd") {
      if (input.dim() == 2) {
        compute_residuals = (rank.item().toInt() == n);
      } else {
        // it is not clear what to do if some matrices have rank < n in case of batched input
        // For now let's compute the residuals only if all matrices have rank equal to n
        // This behaviour may be changed in the future
        // See https://github.com/pytorch/pytorch/issues/56483
        compute_residuals = at::all(rank == n).item().toBool();
      }
    }
    if (compute_residuals) {
      // LAPACK stores residuals data for postprocessing in rows n:(m-n)
      auto raw_residuals = solution.narrow(/*dim=*/-2, /*start=*/n, /*length*/m - n);
      if (raw_residuals.is_complex()) {
        raw_residuals.mul_(raw_residuals.conj());
        raw_residuals = at::real(raw_residuals);
      } else {
        raw_residuals.pow_(2);
      }
      at::sum_out(residuals, raw_residuals, /*dim=*/-2, /*keepdim=*/false, /*dtype*/real_dtype);
    }
  }
  solution = solution.narrow(/*dim=*/-2, /*start=*/0, /*length*/n);
  if (m == 0) {
    solution.zero_();
  }

  // for 1-dimensional 'other', we need to squeeze the solution after "apply_lstsq"
  if (vector_case) {
    solution = solution.squeeze_(-1);
  }
}

static std::string get_default_lstsq_driver(c10::optional<std::string> driver, const Tensor& input) {
  // if `driver` is empty, we set driver_str to "gels" if working with CUDA tensors,
  // otherwise to "gelsy" driver.
  std::string driver_str;
  // check whether the user provided name is a valid driver name
  if (driver.has_value()) {
    driver_str = driver.value();
    // convert `driver_str` to lower case inplace.
    std::transform(driver_str.begin(), driver_str.end(), driver_str.begin(),
      [](unsigned char c) { return std::tolower(c); });
    static std::unordered_set<std::string> allowed_drivers = {
      "gels", "gelsy", "gelsd", "gelss"
    };
    if (input.device() == at::kCPU) {
      TORCH_CHECK(
        allowed_drivers.find(driver_str) != allowed_drivers.end(),
        "torch.linalg.lstsq: parameter `driver` should be one of "
        "(gels, gelsy, gelsd, gelss)"
      );
    } else { // else if (input.is_cuda())
      TORCH_CHECK(
        driver_str == "gels",
        "torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA"
      );
    }
  } else {
    // if driver name is not provided, set to default 'gelsy' if on CPU,
    // or to `gels` if on CUDA.
    driver_str = input.is_cuda() ? "gels" : "gelsy";
  }
  return driver_str;
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> linalg_lstsq_out(
    const Tensor& input,
    const Tensor& other,
    c10::optional<double> rcond,
    c10::optional<std::string> driver,
    Tensor& solution,
    Tensor& residuals,
    Tensor& rank,
    Tensor& singular_values) {
  TORCH_CHECK(input.dim() >= 2, "torch.linalg.lstsq: input must have at least 2 dimensions.");
  TORCH_CHECK(other.dim() >= 1, "torch.linalg.lstsq: other must have at least 1 dimension.");
  TORCH_CHECK(
      input.scalar_type() == other.scalar_type(),
      "torch.linalg.lstsq: Expected input and other to have the same dtype, but got input's dtype ",
      input.scalar_type(),
      " and other's dtype ",
      other.scalar_type());

  auto dim_diff = input.dim() - other.dim();
  TORCH_CHECK(
      0 <= dim_diff && dim_diff <= 1,
      "torch.linalg.lstsq: input.dim() must be greater or equal to other.dim() and (input.dim() - other.dim()) <= 1");
  Tensor other_2d = dim_diff ? other.unsqueeze(-1) : other;
  TORCH_CHECK(
      input.size(-2) == other_2d.size(-2),
      dim_diff ? "torch.linalg.lstsq: input.size(-2) should match other.size(-1)"
               : "torch.linalg.lstsq: input.size(-2) should match other.size(-2)");

  checkSameDevice("torch.linalg.lstsq", other, input, "other");
  checkSameDevice("torch.linalg.lstsq", solution, input, "solution");
  checkSameDevice("torch.linalg.lstsq", residuals, input, "residuals");
  checkSameDevice("torch.linalg.lstsq", rank, input, "rank");
  checkSameDevice("torch.linalg.lstsq", singular_values, input, "singular_values");

  // 'solution' is expected to have same dtype as input
  checkLinalgCompatibleDtype("torch.linalg.lstsq", solution, input, "solution");

  // 'residuals' is expected to have real float dtype
  ScalarType real_dtype = c10::toValueType(input.scalar_type());
  checkLinalgCompatibleDtype("torch.linalg.lstsq", residuals.scalar_type(), real_dtype, "solution");

  // 'rank' is expected to have integer dtype
  // actual LAPACK calls use int32_t type for rank, but we promote it to int64_t
  // to be consistent with torch.linalg.matrix_rank output dtype
  ScalarType rank_expected_type = ScalarType::Long;
  checkLinalgCompatibleDtype("torch.linalg.lstsq", rank.scalar_type(), rank_expected_type, "rank");

  // 'singular_values' is expected to have real float dtype
  checkLinalgCompatibleDtype("torch.linalg.lstsq", singular_values.scalar_type(), real_dtype, "singular_values");

  std::string driver_name = get_default_lstsq_driver(driver, input);

  // set default rcond value
  double rcond_value = rcond.has_value()
    ? rcond.value()
    : _get_epsilon(c10::toValueType(input.scalar_type())) * std::max<int64_t>(input.size(-2), input.size(-1));

  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt));

  // now check whether the provided output tensors can be used directly

  // Two types of 'other' tensors are supported:
  // - 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  // - 2-dimensional (2D) tensor or batch of 2D tensors (matrix case)
  // original torch.lstsq supported only the matrix case, while NumPy works for both cases
  // for the batched input we need to be able to distinguish them
  // auto expected_batched_rhs_shape = IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  // bool vector_case = other.dim() == 1 || (input.dim() - 1 == other.dim() && other.sizes().equals(expected_batched_rhs_shape));
  bool vector_case = linalg_solve_is_vector_rhs(input, other);

  // provided output tensor can be used directly if:
  // 1. the shape matches the expected shape
  // 2. the dtype matches the expected dtype
  // 3. the tensor is contiguous

  // Checks for the 'solution' tensor
  std::vector<int64_t> expected_solution_shape = broadcast_batch_size(input, other_2d, input.dim() - 2);
  // the actual shape of the shape of the solution returned in (*, n,) or (*, n, nrhs)
  // but LAPACK requires extra dimensions so the expected shape is (*, max(m, n),) or (*, max(m, n), nrhs)
  expected_solution_shape.push_back(std::max(input.size(-1), input.size(-2)));
  if (!vector_case && other.dim() > 2) {
    expected_solution_shape.push_back(other.size(-1));
  }

  bool solution_equal_expected_shape = solution.sizes().equals(expected_solution_shape);
  bool solution_input_same_type = (solution.scalar_type() == input.scalar_type());

  bool is_solution_batched_column_major = false;
  if (vector_case) {
    is_solution_batched_column_major = solution.is_contiguous();
  } else if (!vector_case && solution.dim() >= 2) {
    is_solution_batched_column_major = solution.transpose(-2, -1).is_contiguous();
  }

  // 'residuals' is not checked here because at::sum_out(residuals, ...) does that

  auto input_batch_shape = IntArrayRef(input.sizes().cbegin(), input.sizes().cend() - 2);

  // Checks for the 'rank' tensor
  // rank is a scalar value for each matrix in the batch so
  // rank's expected shape is equal to input.shape[0:input.ndim-2]
  bool rank_equal_expected_shape = true;
  bool rank_equal_expected_type = true;
  bool rank_is_contiguous = true;
  if (driver_name != "gels") { // gels driver doesn't set 'rank'
    rank_equal_expected_shape = rank.sizes().equals(input_batch_shape);
    rank_equal_expected_type = (rank.scalar_type() == at::kLong);
    rank_is_contiguous = rank.is_contiguous();
  }

  // Checks for the 'singular_values' tensor
  // singular values are computed only with "gelsd" and "gelss" drivers currently
  bool singular_values_equal_expected_shape = true;
  bool singular_values_equal_expected_type = true;
  bool singular_values_is_contiguous = true;
  if (driver_name == "gelsd" || driver_name == "gelss") {
    auto singular_values_shape = input_batch_shape.vec();
    singular_values_shape.push_back(std::min(input.size(-1), input.size(-2)));
    singular_values_equal_expected_shape = singular_values.sizes().equals(singular_values_shape);
    singular_values_equal_expected_type = (singular_values.scalar_type() == real_dtype);
    singular_values_is_contiguous = singular_values.is_contiguous();
  }

  // if solution is not empty and not in batched column major format
  bool copy_needed = (solution.numel() != 0 && !is_solution_batched_column_major);
  copy_needed |= !solution_input_same_type;  // or solution does not have the same dtype as input
  copy_needed |= (solution.numel() != 0 && !solution_equal_expected_shape); // or solution does not have the expected shape

  copy_needed |= !rank_equal_expected_type;
  copy_needed |= (rank.numel() != 0 && !rank_equal_expected_shape);
  copy_needed |= (rank.numel() != 0 && !rank_is_contiguous);

  copy_needed |= !singular_values_equal_expected_type;
  copy_needed |= (singular_values.numel() != 0 && !singular_values_equal_expected_shape);
  copy_needed |= (singular_values.numel() != 0 && !singular_values_is_contiguous);

  if (copy_needed) { // we have to allocate temporary tensors
    Tensor solution_tmp = at::empty({0}, input.options());
    Tensor residuals_tmp = at::empty({0}, input.options().dtype(real_dtype));
    Tensor rank_tmp = at::empty({0}, input.options().dtype(at::kLong));
    Tensor singular_values_tmp = at::empty({0}, input.options().dtype(real_dtype));

    linalg_lstsq_out_info(solution_tmp, residuals_tmp, rank_tmp, singular_values_tmp, infos, input, other, rcond_value, driver_name);

    at::native::resize_output(solution, solution_tmp.sizes());
    solution.copy_(solution_tmp);

    at::native::resize_output(residuals, residuals_tmp.sizes());
    residuals.copy_(residuals_tmp);

    at::native::resize_output(rank, rank_tmp.sizes());
    rank.copy_(rank_tmp);

    at::native::resize_output(singular_values, singular_values_tmp.sizes());
    singular_values.copy_(singular_values_tmp);
  } else {
    // else use the provided output storage directly
    linalg_lstsq_out_info(solution, residuals, rank, singular_values, infos, input, other, rcond_value, driver_name);
  }

  if (infos.numel() > 1) {
    batchCheckErrors(infos, "torch.linalg.lstsq");
  } else {
    singleCheckErrors(infos.item<int64_t>(), "torch.linalg.lstsq");
  }

  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(solution, residuals, rank, singular_values);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq(
    const Tensor& input, const Tensor& other,
    c10::optional<double> rcond,
    c10::optional<std::string> driver) {
  Tensor solution = at::empty({0}, input.options());
  Tensor residuals = at::empty({0}, input.options().dtype(toValueType(input.scalar_type())));
  Tensor rank = at::empty({0}, input.options().dtype(at::kLong));
  Tensor singular_values = at::empty({0}, input.options().dtype(toValueType(input.scalar_type())));
  std::tie(solution, residuals, rank, singular_values) =
      at::linalg_lstsq_outf(input, other, rcond, driver, solution, residuals, rank, singular_values);
  return std::make_tuple(solution, residuals, rank, singular_values);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lu_solve_stub);

// Supports arbitrary batch dimensions for self and LU_data (implicitly LU_pivots also)
Tensor lu_solve(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  TORCH_CHECK(self.dim() >= 2,
              "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  TORCH_CHECK(LU_data.dim() >= 2,
              "LU_data should have at least 2 dimensions, but has ", LU_data.dim(), " dimensions instead");
  TORCH_CHECK(LU_pivots.size(-1) == LU_data.size(-1),
              "Number of pivots per batch should be same as the dimension of the matrix");
  TORCH_CHECK(LU_pivots.dtype() == at::kInt,
              "LU_pivots should be a Tensor of scalar type Int");
  TORCH_CHECK(LU_pivots.device() == LU_data.device(),
              "Expected LU_pivots and LU_data to be on the same device, "
              "but found LU_pivots on ", LU_pivots.device(), " and LU_data on ",
              LU_data.device(), " instead");

  // We check whether the batch dimensions of LU_pivots match the batch dimensions of LU_data
  // e.g.: LU_pivots.sizes() = 4 x 3 x 2, LU_data.sizes() = 4 x 3 x 2 x 2 is a pair of correct inputs
  // e.g.: LU_pivots.sizes() = 4 x 3 x 2, LU_data.sizes() = 12 x 2 x 2 is a pair of incorrect inputs
  IntArrayRef pivots_sizes(LU_pivots.sizes().data(), LU_pivots.dim() - 1);
  IntArrayRef lu_sizes(LU_data.sizes().data(), LU_data.dim() - 2);
  TORCH_CHECK(pivots_sizes == lu_sizes,
              "batch dimensions of LU_pivots doesn't match batch dimensions of LU_data");

  Tensor self_broadcasted, LU_data_broadcasted;
  std::tie(self_broadcasted, LU_data_broadcasted) = _linalg_broadcast_batch_dims(self, LU_data, "lu_solve");

  // Now, we need to broadcast pivots too for the batch dimensions to match
  IntArrayRef new_pivots_sizes(LU_data_broadcasted.sizes().data(), LU_data_broadcasted.dim() - 1);
  Tensor LU_pivots_broadcasted = LU_pivots.expand(new_pivots_sizes);

  // lu_solve_stub (apply_lu_solve) requires batched column major (Fortran-contiguous) tensors
  // 'result' tensor is modified in-place and must be a copy of 'self_broadcasted'
  Tensor result = cloneBatchedColumnMajor(self_broadcasted);

  // if LU_data is Fortran-contiguous no need to make a copy
  bool is_LU_data_batched_column_major = LU_data_broadcasted.transpose(-2, -1).is_contiguous();
  Tensor LU_data_working_copy = is_LU_data_batched_column_major ? LU_data_broadcasted : cloneBatchedColumnMajor(LU_data_broadcasted);
  Tensor LU_pivots_working_copy = LU_pivots_broadcasted.is_contiguous() ? LU_pivots_broadcasted : LU_pivots_broadcasted.contiguous();

  lu_solve_stub(self.device().type(), result, LU_data_working_copy, LU_pivots_working_copy);
  return result;
}

Tensor& lu_solve_out(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots, Tensor& result) {
  checkSameDevice("lu_solve", result, self);
  checkLinalgCompatibleDtype("lu_solve", result, self);
  Tensor result_tmp = at::lu_solve(self, LU_data, LU_pivots);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

}}  // namespace at::native
