#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper.h>
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/_linalg_eigh.h>
#include <ATen/ops/_linalg_eigh_meta.h>
#include <ATen/ops/_linalg_eigh_native.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_linalg_solve_ex_meta.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/_lu_with_info_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cholesky.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/cholesky_inverse_native.h>
#include <ATen/ops/cholesky_native.h>
#include <ATen/ops/cholesky_solve.h>
#include <ATen/ops/cholesky_solve_native.h>
#include <ATen/ops/clone.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/cumprod.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/geqrf.h>
#include <ATen/ops/geqrf_native.h>
#include <ATen/ops/inverse_native.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cholesky_ex_meta.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_cholesky_native.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eig_native.h>
#include <ATen/ops/linalg_eigh_native.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvals_native.h>
#include <ATen/ops/linalg_eigvalsh_native.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_householder_product_native.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_inv_native.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_factor_ex_meta.h>
#include <ATen/ops/linalg_ldl_factor_ex_native.h>
#include <ATen/ops/linalg_ldl_factor_native.h>
#include <ATen/ops/linalg_ldl_solve_meta.h>
#include <ATen/ops/linalg_ldl_solve_native.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_factor_ex_meta.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_qr_meta.h>
#include <ATen/ops/linalg_qr_native.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_ex_native.h>
#include <ATen/ops/linalg_solve_native.h>
#include <ATen/ops/linalg_solve_triangular_native.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svd_native.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_svdvals_native.h>
#include <ATen/ops/linalg_vander_native.h>
#include <ATen/ops/linalg_vecdot_native.h>
#include <ATen/ops/lu_solve_native.h>
#include <ATen/ops/lu_unpack.h>
#include <ATen/ops/lu_unpack_meta.h>
#include <ATen/ops/lu_unpack_native.h>
#include <ATen/ops/orgqr_native.h>
#include <ATen/ops/ormqr_native.h>
#include <ATen/ops/qr_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/svd_native.h>
#include <ATen/ops/triangular_solve_meta.h>
#include <ATen/ops/triangular_solve_native.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/ops/vdot.h>
#include <ATen/ops/zeros.h>
#endif

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#if AT_BUILD_WITH_LAPACK()

// getrf
extern "C" void zgetrf_(int *m, int *n, std::complex<double> *a, int *lda, int *ipiv, int *info);
extern "C" void cgetrf_(int *m, int *n, std::complex<float> *a, int *lda, int *ipiv, int *info);
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);

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

// sytrf
extern "C" void dsytrf_(
    char* uplo,
    int* n,
    double* a,
    int* lda,
    int* ipiv,
    double* work,
    int* lwork,
    int* info);
extern "C" void ssytrf_(
    char* uplo,
    int* n,
    float* a,
    int* lda,
    int* ipiv,
    float* work,
    int* lwork,
    int* info);
extern "C" void zsytrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void csytrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// hetrf
extern "C" void zhetrf_(
    char* uplo,
    int* n,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* work,
    int* lwork,
    int* info);
extern "C" void chetrf_(
    char* uplo,
    int* n,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* work,
    int* lwork,
    int* info);

// sytrs
extern "C" void dsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    double* a,
    int* lda,
    int* ipiv,
    double* b,
    int* ldb,
    int* info);
extern "C" void ssytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    float* a,
    int* lda,
    int* ipiv,
    float* b,
    int* ldb,
    int* info);
extern "C" void zsytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void csytrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

// hetrs
extern "C" void zhetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<double>* a,
    int* lda,
    int* ipiv,
    std::complex<double>* b,
    int* ldb,
    int* info);
extern "C" void chetrs_(
    char* uplo,
    int* n,
    int* nrhs,
    std::complex<float>* a,
    int* lda,
    int* ipiv,
    std::complex<float>* b,
    int* ldb,
    int* info);

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

#if AT_BUILD_WITH_BLAS()
// trsm
extern "C" void ztrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<double> *alpha, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb);
extern "C" void ctrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, std::complex<float> *alpha, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb);
extern "C" void dtrsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, double *alpha, double *a, int *lda, double *b, int *ldb);
extern "C" void strsm_(char *side, char *uplo, char *trans, char *diag, int *n, int *nrhs, float *alpha, float *a, int *lda, float *b, int *ldb);
#endif

namespace at {
namespace meta {

TORCH_META_FUNC(linalg_ldl_factor_ex)
(const Tensor& self, bool hermitian, bool check_errors) {
  at::native::squareCheckInputs(self, "torch.linalg.ldl_factor_ex");
  at::native::checkFloatingOrComplex(self, "torch.linalg.ldl_factor_ex");

  auto shape = self.sizes();
  auto ndim = shape.size();

  // prefer column major strides
  auto ld_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig=*/true);
  set_output_strided(0, shape, ld_strides, self.options(), {}); // LD

  set_output_contiguous(
      1, shape.slice(0, ndim - 1), self.options().dtype(ScalarType::Int)); // pivots

  set_output_contiguous(
      2, shape.slice(0, ndim - 2), self.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_ldl_solve)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian) {
  at::native::squareCheckInputs(LD, "torch.linalg.ldl_solve");
  at::native::checkFloatingOrComplex(LD, "torch.linalg.ldl_solve");
  at::native::linearSolveCheckInputs(B, LD, "torch.linalg.ldl_solve");
  TORCH_CHECK(
      B.dim() >= 2,
      "torch.linalg.ldl_solve: Expected B to have at least 2 dimensions, but it has ",
      B.dim(),
      " dimensions instead");
  auto expected_pivots_shape = LD.sizes().slice(0, LD.dim() - 1);
  TORCH_CHECK(
      expected_pivots_shape.equals(pivots.sizes()),
      "torch.linalg.ldl_solve: Expected LD.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(),
      " instead");
  // pivots is allowed to be any integer type
  // LAPACK we use is 32-bit interface while cuSOLVER uses 64-bit interface for integers
  TORCH_CHECK(
      at::isIntegralType(pivots.scalar_type(), /*includeBool=*/false),
      "torch.linalg.ldl_solve: Expected pivots to be integers. Got ",
      pivots.scalar_type());
  TORCH_CHECK(
      LD.scalar_type() == B.scalar_type(),
      "torch.linalg.ldl_solve: ",
      "LD dtype",
      LD.scalar_type(),
      " does not match b dtype ",
      B.scalar_type());

    std::vector<int64_t> B_broadcast_size;
    std::tie(B_broadcast_size, std::ignore) = at::native::_linalg_broadcast_batch_dims(B, LD);

  // prefer column major strides
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/true);
  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(triangular_solve)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular) {
  TORCH_CHECK(self.dim() >= 2,
           "torch.triangular_solve: Expected b to have at least 2 dimensions, but it has ", self.dim(), " dimensions instead");
  TORCH_CHECK(A.dim() >= 2,
           "torch.triangular_solve: Expected A to have at least 2 dimensions, but it has ", A.dim(), " dimensions instead");

  at::native::linearSolveCheckInputs(self, A, "triangular_solve");

  if (A.layout() == Layout::Strided) {
    std::vector<int64_t> self_broadcast_size, A_broadcast_size;
    std::tie(self_broadcast_size, A_broadcast_size) = at::native::_linalg_broadcast_batch_dims(self, A);

    // make column major strides for BLAS
    const auto solution_strides = at::native::batched_matrix_contiguous_strides(self_broadcast_size, /*f-contig=*/true);
    set_output_raw_strided(0, self_broadcast_size, solution_strides, self.options(), {});

    // make column major strides for BLAS
    auto clone_A_strides = at::native::batched_matrix_contiguous_strides(A_broadcast_size, /*f_contig=*/true);
    set_output_raw_strided(1, A_broadcast_size, clone_A_strides, A.options(), {});
  } else if (A.layout() == Layout::SparseCsr || A.layout() == Layout::SparseBsr) {
    // no broadcasting for non-strided layout
    set_output_raw_strided(0, self.sizes(), {}, self.options(), {}); // make row major strides for Sparse BLAS
    set_output_raw_strided(1, {0}, {}, self.options(), {}); // return 0-sized tensor
  } else {
    TORCH_INTERNAL_ASSERT(false, "triangular_solve: Got an unexpected layout.");
  }
}

TORCH_META_FUNC(_linalg_solve_ex)(const Tensor& A,
                                  const Tensor& B,
                                  bool left,
                                  bool check_errors) {
  // dtype
  at::native::checkFloatingOrComplex(A, "linalg.solve");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(),
              "linalg.solve: Expected A and B to have the same dtype, but found A of type ",
              A.scalar_type(), " and B of type ", B.scalar_type(), " instead");

  // NumPy compat: Two types of 'B' tensors are supported:
  // - 1D tensor or batch of 1D tensors (vector case)
  // - 2D tensor or batch of 2D tensors (matrix case)
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(A, B);
  auto B_ = vector_case ? B.unsqueeze(-1) : B;

  // matrix shapes
  at::native::checkInputsSolver(A, B_, /*left=*/left, "linalg.solve");

  // Check that B can be broadcasted to the shape of A
  auto B_broad_shape = std::get<0>(at::native::_linalg_broadcast_batch_dims(B_, A));
  // We disallow the broadcasting of B as a vector when left=False as, in that case, A.shape = (*, 1, 1)
  TORCH_CHECK(left || !vector_case, "linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)");
  auto result_shape = vector_case ? IntArrayRef(B_broad_shape.data(), B_broad_shape.size() - 1)
                                  : B_broad_shape;
  auto result_strides = at::native::batched_matrix_contiguous_strides(result_shape, /*column_major=*/left);

  set_output_strided(0, result_shape, result_strides, B.options(), {});

  auto shape = A.sizes();
  auto ndim = shape.size();

  // LU
  auto LU_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(1, shape, LU_strides, A.options(), {});

  // pivots
  set_output_contiguous(2, shape.slice(0, ndim - 1), A.options().dtype(kInt));

  // info
  set_output_contiguous(3, shape.slice(0, ndim - 2), A.options().dtype(kInt));
}

TORCH_META_FUNC(linalg_inv_ex)(const Tensor& A, bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.inv");
  at::native::checkFloatingOrComplex(A, "linalg.inv", /*allow_low_precision_dtypes*/false);

  auto shape = A.sizes();

  auto result_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  set_output_strided(0, shape, result_strides, A.options(), {});
  set_output_contiguous(
      1, shape.slice(0, shape.size() - 2), A.options().dtype(ScalarType::Int)); // info
}

TORCH_META_FUNC(linalg_lu_factor_ex)(const Tensor& A, bool pivot, bool check_errors) {
  TORCH_CHECK(A.dim() >= 2, "torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];

  // make column major strides for BLAS
  auto LU_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
  set_output_strided(0, sizes, LU_strides, A.options(), {});

  // Set sizes to the size of pivots
  sizes.pop_back();
  sizes.back() = std::min(m, n);
  set_output_contiguous(1, sizes, A.options().dtype(kInt), {});

  // Set sizes to the size of info
  sizes.pop_back();
  set_output_contiguous(2, sizes, A.options().dtype(kInt), {});
}

TORCH_META_FUNC(linalg_lu_solve)(const Tensor& LU,
                                 const Tensor& pivots,
                                 const Tensor& B,
                                 bool left,
                                 bool adjoint) {
  // dtype
  at::native::checkFloatingOrComplex(LU, "torch.linalg.lu_solve");
  TORCH_CHECK(LU.scalar_type() == B.scalar_type(),
              "linalg.lu_solve: Expected LU and B to have the same dtype, but found LU of type ",
              LU.scalar_type(), " and B of type ", B.scalar_type(), " instead");
  TORCH_CHECK(pivots.dtype() == at::kInt,
              "linalg.lu_solve: pivots should be a Tensor of scalar type torch.int32");

  // matrix shapes
  at::native::squareCheckInputs(LU, "torch.linalg.lu_solve");
  at::native::checkInputsSolver(LU, B, left, "linalg.lu_solve");
  //
  TORCH_CHECK(LU.size(-1) == pivots.size(-1),
              "linalg.lu_solve: Number of pivots per batch should be same as the dimension of the matrix");

  // batches
  TORCH_CHECK(
      LU.sizes().slice(0, LU.dim() - 1).equals(pivots.sizes()),
      "linalg.lu_solve: Expected LU.shape[:-1] and pivots.shape to be the same, but got pivots with shape ",
      pivots.sizes(), " instead");

  // This one checks that B can be broadcasted to the shape of A
  auto B_broadcast_size = std::get<0>(at::native::_linalg_broadcast_batch_dims(B, LU));
  auto result_strides = at::native::batched_matrix_contiguous_strides(B_broadcast_size, /*column_major=*/left);

  set_output_strided(0, B_broadcast_size, result_strides, B.options(), {});
}

TORCH_META_FUNC(linalg_cholesky_ex)(const Tensor& A,
                                    bool upper,
                                    bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.cholesky");
  at::native::checkFloatingOrComplex(A, "linalg.cholesky");

  auto A_shape = A.sizes();
  auto ndim = A_shape.size();

  // L
  auto L_strides = at::native::batched_matrix_contiguous_strides(A_shape, /*f-contig*=*/true);
  set_output_strided(0, A_shape, L_strides, A.options(), {});

  // info
  set_output_contiguous(1, A_shape.slice(0, ndim - 2), A.options().dtype(ScalarType::Int));
}

TORCH_META_FUNC(linalg_qr)(const Tensor& A,
                           c10::string_view mode) {
  at::native::checkIsMatrix(A, "linalg.qr");
  at::native::checkFloatingOrComplex(A, "linalg.qr");
  bool compute_q, reduced_mode;
  std::tie(compute_q, reduced_mode) = at::native::_parse_qr_mode(mode);

  auto A_shape = A.sizes().vec();
  const auto m = A_shape.cend()[-2];
  const auto n = A_shape.cend()[-1];
  const auto k = std::min(m, n);

  if (compute_q) {
    auto Q_shape = A_shape;
    Q_shape.end()[-1] = reduced_mode ? k : m;
    auto Q_strides = at::native::batched_matrix_contiguous_strides(Q_shape, /*f-contig*=*/true);
    set_output_strided(0, Q_shape, Q_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // For readability
  auto R_shape = std::move(A_shape);
  R_shape.end()[-2] = (reduced_mode || !compute_q) ? k : m;
  auto R_strides = at::native::batched_matrix_contiguous_strides(R_shape, /*f-contig*=*/true);
  set_output_strided(1, R_shape, R_strides, A.options(), {});
}


TORCH_META_FUNC(_linalg_svd)(const Tensor& A,
                             bool full_matrices,
                             bool compute_uv,
                             c10::optional<c10::string_view> driver) {
  at::native::checkIsMatrix(A, "linalg.svd");
  at::native::checkFloatingOrComplex(A, "linalg.svd");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // Prepare sizes for U
  if (compute_uv) {
    sizes.back() = full_matrices ? m : k;
    auto U_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
    set_output_strided(0, sizes, U_strides, A.options(), {});

    // Prepare sizes for Vh
    sizes.end()[-2] = full_matrices ? n : k;
    sizes.end()[-1] = n;

    // We need to distinguish the cuSOLVER case, as the cuSOLVER algorithms we use
    // expect F-contig matrices, but they compute V rather than Vh
    const bool use_cusolver = at::native::svd_uses_cusolver(A);
    auto Vh_strides = at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/!use_cusolver);
    set_output_strided(2, sizes, Vh_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
    set_output_raw_strided(2, {0}, {}, A.options(), {});
  }

  // Prepare sizes for S. S is always real, even when A is complex.
  sizes.pop_back();
  sizes.end()[-1] = k;
  set_output_contiguous(1, sizes, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(lu_unpack)(const Tensor& LU, const Tensor& pivots, bool unpack_data, bool unpack_pivots) {
  TORCH_CHECK(LU.dim() >= 2, "torch.lu_unpack: Expected tensor with 2 or more dimensions. Got size: ", LU.sizes(), " instead");
  if (unpack_pivots) {
    TORCH_CHECK(pivots.scalar_type() == at::kInt,
        "torch.lu_unpack: LU_pivots is expected to be a contiguous tensor of torch.int32 dtype.\n"
        "Note: this function is intended to be used with the output produced by torch.linalg.lu_factor");
  }

  auto sizes = LU.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (unpack_pivots) {
    set_output_raw_strided(0, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, LU.options(), {});
  }

  if (unpack_data) {
    // L.shape[-2:] == (m, k)
    sizes.end()[-1] = k;
    set_output_raw_strided(1, sizes, {}, LU.options(), {});

    // U.shape[-2:] == (k, n)
    sizes.end()[-2] = k;
    sizes.end()[-1] = n;
    set_output_raw_strided(2, sizes, {}, LU.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, LU.options(), {});
    set_output_raw_strided(2, {0}, {}, LU.options(), {});
  }
}

TORCH_META_FUNC(_linalg_eigh)(const Tensor& A,
                              c10::string_view uplo,
                              bool compute_v) {
  at::native::squareCheckInputs(A, "linalg.eigh");
  at::native::checkUplo(uplo);

  auto shape = A.sizes().vec();
  if (compute_v) {
    // eigenvectors
    auto V_strides = at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
    set_output_strided(1, shape, V_strides, A.options(), {});
  } else {
    set_output_raw_strided(1, {0}, {}, A.options(), {});
  }

  // eigenvalues
  shape.pop_back();
  set_output_contiguous(0, shape, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_META_FUNC(linalg_lu)(const Tensor& A, bool pivot) {
  TORCH_CHECK(A.dim() >= 2, "linalg.lu: Expected tensor with 2 or more dimensions. Got size: ", A.sizes(), " instead");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // P.shape[-2:] == (m, m) (or size zero if pivot == False)
  sizes.end()[-1] = m;
  if (pivot) {
    set_output_raw_strided(0, sizes, {}, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
  }

  // L.shape[-2:] == (m, k)
  sizes.end()[-1] = k;
  set_output_raw_strided(1, sizes, {}, A.options(), {});

  // U.shape[-2:] == (k, n)
  sizes.end()[-2] = k;
  sizes.end()[-1] = n;
  set_output_raw_strided(2, sizes, {}, A.options(), {});
}

} // namespace meta

namespace native {

#if AT_BUILD_WITH_LAPACK()
// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations

template<class scalar_t>
void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info);

template<class scalar_t, class value_t=scalar_t>
void lapackSymeig(char jobz, char uplo, int n, scalar_t *a, int lda, value_t *w, scalar_t *work, int lwork, value_t *rwork, int *info);

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

template <>
void lapackLdlSymmetric<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  zsytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  csytrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<double>(
    char uplo,
    int n,
    double* a,
    int lda,
    int* ipiv,
    double* work,
    int lwork,
    int* info) {
  dsytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<float>(
    char uplo,
    int n,
    float* a,
    int lda,
    int* ipiv,
    float* work,
    int lwork,
    int* info) {
  ssytrf_(&uplo, &n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void lapackLdlHermitian<c10::complex<double>>(
    char uplo,
    int n,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  zhetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlHermitian<c10::complex<float>>(
    char uplo,
    int n,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  chetrf_(
      &uplo,
      &n,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(work),
      &lwork,
      info);
}

template <>
void lapackLdlSolveSymmetric<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveSymmetric<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  zsytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveSymmetric<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  csytrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<double>(
    char uplo,
    int n,
    int nrhs,
    double* a,
    int lda,
    int* ipiv,
    double* b,
    int ldb,
    int* info) {
  dsytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<float>(
    char uplo,
    int n,
    int nrhs,
    float* a,
    int lda,
    int* ipiv,
    float* b,
    int ldb,
    int* info) {
  ssytrs_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<double>* a,
    int lda,
    int* ipiv,
    c10::complex<double>* b,
    int ldb,
    int* info) {
  zhetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      &ldb,
      info);
}

template <>
void lapackLdlSolveHermitian<c10::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    c10::complex<float>* a,
    int lda,
    int* ipiv,
    c10::complex<float>* b,
    int ldb,
    int* info) {
  chetrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      &lda,
      ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      &ldb,
      info);
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

#if AT_BUILD_WITH_BLAS()
template<> void blasTriangularSolve<c10::complex<double>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<double> *a, int lda, c10::complex<double> *b, int ldb) {
  std::complex<double> one{1., 0.};
  ztrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<double>*>(a), &lda, reinterpret_cast<std::complex<double>*>(b), &ldb);
}

template<> void blasTriangularSolve<c10::complex<float>>(char side, char uplo, char trans, char diag, int n, int nrhs, c10::complex<float> *a, int lda, c10::complex<float> *b, int ldb) {
  std::complex<float> one{1.f, 0.f};
  ctrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, reinterpret_cast<std::complex<float>*>(a), &lda, reinterpret_cast<std::complex<float>*>(b), &ldb);
}

template<> void blasTriangularSolve<double>(char side, char uplo, char trans, char diag, int n, int nrhs, double *a, int lda, double *b, int ldb) {
  auto one = 1.;
  dtrsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}

template<> void blasTriangularSolve<float>(char side, char uplo, char trans, char diag, int n, int nrhs, float *a, int lda, float *b, int ldb) {
  auto one = 1.f;
  strsm_(&side, &uplo, &trans, &diag, &n, &nrhs, &one, a, &lda, b, &ldb);
}
#endif

void _linalg_check_errors(
    const Tensor& infos,
    const c10::string_view api_name,
    bool is_matrix) {
  TORCH_INTERNAL_ASSERT(infos.scalar_type() == kInt);
  TORCH_INTERNAL_ASSERT(infos.is_contiguous());
  if (infos.is_meta()) {
    return;
  }

  // If it's all zeros, we return early.
  // We optimise for the most likely case.
  if (C10_LIKELY(!infos.any().item<bool>())) {
    return;
  }

  int32_t info;
  std::string batch_str;
  if (is_matrix) {
    info = infos.item<int>();
    // batch_str needn't be set for matrices
  } else {
    // Find the first non-zero info
    auto infos_cpu = infos.to(at::kCPU);
    auto ptr = infos_cpu.data_ptr<int32_t>();
    auto n = infos.numel();
    auto info_ptr = std::find_if(ptr, ptr + n, [](int32_t x) { return x != 0; });
    info = *info_ptr;
    batch_str = ": (Batch element " + std::to_string(std::distance(ptr, info_ptr)) + ")";
  }

  if (info < 0) {
    // Reference LAPACK 3.10+ changed `info` behavior for inputs with non-finite values
    // Previously, it would return `info` > 0, but now it returns `info` = -4
    // OpenBLAS 0.3.15+ uses the Reference LAPACK 3.10+.
    // MKL 2022.0+ uses the Reference LAPACK 3.10+.
    // Older version of MKL and OpenBLAS follow the old behavior (return `info` > 0).
    // Here we check for the case where `info` is -4 and raise an error
    if (api_name.find("svd") != api_name.npos) {
      TORCH_CHECK_LINALG(info != -4, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix contained non-finite values.");
    }
    TORCH_INTERNAL_ASSERT(false, api_name, batch_str,
        ": Argument ", -info, " has illegal value. Most certainly there is a bug in the implementation calling the backend library.");
  } else if (info > 0) {
    if (api_name.find("inv") != api_name.npos) {
      // inv, inverse, cholesky_inverse, etc.
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (api_name.find("solve") != api_name.npos) {
      // solve, linalg_solve, cholesky_solve, etc.
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The solver failed because the input matrix is singular.");
    } else if (api_name.find("cholesky") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (api_name.find("svd") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (api_name.find("eig") != api_name.npos || api_name.find("syevd") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else if (api_name.find("lstsq") != api_name.npos) {
      TORCH_CHECK_LINALG(false, api_name, batch_str,
          ": The least squares solution could not be computed because the input matrix does not have full rank (error code: ", info, ").");
    } else if (api_name.find("lu_factor") != api_name.npos) {
      TORCH_CHECK(false, api_name, batch_str,
          ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
          "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
          "linalg.lu_factor_ex(A, pivot)");
    } else {
      TORCH_INTERNAL_ASSERT(false, api_name, ": Unknown error code: ", info, ".");
    }
  }
  // We should never reach this point as info was non-zero
  TORCH_INTERNAL_ASSERT(false);
}

// If an input requires fw or bw grad then we need to go down a different
// (slower) path to ensure that the gradients are computable.
// That is what `_may_require_fw_or_bw_grad` is helpful for.
//
// Why is there a isTensorSubclassLike check here?
// Without it, this function can lead to composite compliance problems, which
// may lead to bugs in functorch, where a Tensor Subclass that doesn't
// require grad may wrap a Tensor subclass that requires grad.
static bool _may_require_fw_or_bw_grad(const Tensor& input) {
  return ((at::GradMode::is_enabled() && input.requires_grad())
          || input._fw_grad(/*level */ 0).defined()
          || isTensorSubclassLike(input));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.inv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TORCH_IMPL_FUNC(linalg_inv_ex_out)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info) {
  // Fill result with the identity
  result.zero_();
  result.diagonal(0, -2, -1).fill_(1.);
  at::linalg_solve_ex_out(const_cast<Tensor&>(result), const_cast<Tensor&>(info), A, result, /*left*/true);
  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.inv_ex", A.dim() == 2);
  }
}

Tensor& linalg_inv_out(const Tensor& A, Tensor& result) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_inv_ex_out(result, info, A);
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

Tensor linalg_inv(const Tensor& A) {
  Tensor result, info;
  std::tie(result, info) = at::linalg_inv_ex(A);
  at::_linalg_check_errors(info, "linalg.inv", A.dim() == 2);
  return result;
}

Tensor& inverse_out(const Tensor& A, Tensor& result) {
  return at::linalg_inv_out(result, A);
}

Tensor inverse(const Tensor& A) {
  return at::linalg_inv(A);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, Tensor& infos) {
#if !AT_BUILD_WITH_LAPACK()
  AT_ERROR("cholesky_solve: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto ldab = std::max<int64_t>(1, n);
  auto nrhs = b.size(-1);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    lapackCholeskySolve<scalar_t>(uplo, n, nrhs, A_working_ptr, ldab, b_working_ptr, ldab, &info);
    infos_data[i] = info;
    if (info != 0) {
      return;
    }
  }
#endif
}

Tensor _cholesky_solve_helper_cpu(const Tensor& self, const Tensor& A, bool upper) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto infos = at::zeros({batchCount(self)}, self.options().dtype(kInt));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cpu", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, infos);
  });

  at::_linalg_check_errors(infos, "cholesky_solve_cpu", self.dim() == 2);
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
    "U = torch.linalg.cholesky(A).mH().\n"
    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  if (self.numel() == 0) {
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  squareCheckInputs(self, "cholesky");

  auto raw_cholesky_output = cloneBatchedColumnMajor(self);
  auto info_shape = IntArrayRef(
      self.sizes().cbegin(), self.sizes().cend() - 2); // self.shape[:-2]
  auto info = at::empty({info_shape}, self.options().dtype(kInt));

  // fill the raw_cholesky_output with the result
  cholesky_stub(self.device().type(), raw_cholesky_output, info, upper);

  at::_linalg_check_errors(info, "cholesky", self.dim() == 2);

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
    "U = torch.linalg.cholesky(A).mH().\n"
    "This transform will produce equivalent results for all valid (symmetric positive definite) inputs."
  );
  checkSameDevice("cholesky", result, self);
  checkLinalgCompatibleDtype("cholesky", result, self);
  Tensor result_tmp = at::cholesky(self, upper);
  at::native::resize_output(result, result_tmp.sizes());
  result.copy_(result_tmp);
  return result;
}

TORCH_IMPL_FUNC(linalg_cholesky_ex_out)(const Tensor& A,
                                        bool upper,
                                        bool check_errors,
                                        const Tensor& L,
                                        const Tensor& info) {
  // Nothing to do there
  if (L.numel() == 0) {
    info.zero_();
    return;
  }
  const auto cpu = A.device() == kCPU;

  // We can perform this optimisation just on CPU as it fails for MAGMA
  // due to some bug
  if (cpu) {
    if (upper) {
      at::triu_out(const_cast<Tensor&>(L), A);
    } else {
      at::tril_out(const_cast<Tensor&>(L), A);
    }
  } else {
    L.copy_(A);
  }

  cholesky_stub(L.device().type(), L, info, upper);

  if (!cpu) {
    if (upper) {
      L.triu_();
    } else {
      L.tril_();
    }
  }

  if (check_errors) {
    at::_linalg_check_errors(info, "linalg.cholesky_ex", A.dim() == 2);
  }
}

Tensor linalg_cholesky(const Tensor& A, bool upper) {
  Tensor L, info;
  std::tie(L, info) = at::linalg_cholesky_ex(A, upper, /*check_errors=*/false);
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  return L;
}

Tensor& linalg_cholesky_out(const Tensor& A, bool upper, Tensor& L) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_cholesky_ex_out(L, info, A, upper, /*check_errors=*/false);
  at::_linalg_check_errors(info, "linalg.cholesky", A.dim() == 2);
  return L;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(cholesky_inverse_stub);

static Tensor& cholesky_inverse_out_info(Tensor& result, Tensor& infos, const Tensor& input, bool upper) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-1) == input.size(-2));

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  TORCH_INTERNAL_ASSERT(infos.scalar_type() == at::kInt);
  TORCH_INTERNAL_ASSERT(infos.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT(infos.numel() == std::max<int64_t>(1, batchCount(input)));

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
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
  squareCheckInputs(input, "cholesky_inverse");
  checkSameDevice("cholesky_inverse", result, input);
  checkLinalgCompatibleDtype("cholesky_inverse", result, input);

  // MAGMA requires 'infos' to reside in CPU memory, therefore we create 'infos' only on CPU for now.
  auto infos = at::zeros({std::max<int64_t>(1, batchCount(input))}, input.options().dtype(kInt).device(kCPU));

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.mT().is_contiguous();
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
  at::_linalg_check_errors(infos, "cholesky_inverse", result.dim() == 2);
  return result;
}

Tensor cholesky_inverse(const Tensor &input, bool upper) {
  Tensor result = at::empty({0}, input.options());
  result = at::cholesky_inverse_out(result, input, upper);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg.solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Auxiliary function that returns the LU decomposition to use it in the backward
TORCH_IMPL_FUNC(_linalg_solve_ex_out)(const Tensor& A,
                                      const Tensor& B,
                                      bool left,
                                      bool check_errors,
                                      const Tensor& result,
                                      const Tensor& LU,
                                      const Tensor& pivots,
                                      const Tensor& info) {
  // Possible optimization: Compute the LU factorization of A^T if A is contiguous
  // Then we solve A^T X = B with adjoint=True
  // This saves a copy as A doesn't need to be copied into an F-contig matrix in lu_factor
  // This optimization makes functorch's batching rule difficult. See NOTE [ solve_ex Batch Rule Contiguity ]
  const bool use_A_T = A.is_contiguous() && !A.is_complex();
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(LU),
                              const_cast<Tensor&>(pivots),
                              const_cast<Tensor&>(info),
                              use_A_T ? A.mT() : A);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", A.dim() == 2);
  }

  // [numpy-compat] Handle vectors on the rhs
  const bool vector_case = at::native::linalg_solve_is_vector_rhs(LU, B);
  auto result_ = vector_case ? result.unsqueeze(-1) : result;
  auto B_ = vector_case ? B.unsqueeze(-1) : B;
  at::linalg_lu_solve_out(result_, LU, pivots, B_, left, /*adjoint*/use_A_T);
}

std::tuple<Tensor&, Tensor&> linalg_solve_ex_out(const Tensor& A,
                                                 const Tensor& B,
                                                 bool left,
                                                 bool check_errors,
                                                 Tensor& result,
                                                 Tensor& info) {
  auto LU = B.new_empty({0});
  auto pivots = B.new_empty({0}, kInt);
  at::_linalg_solve_ex_out(result, LU, pivots, info, A, B, left, check_errors);
  return std::tie(result, info);
}

// We implement linalg_solve_ex as a composite function of _linalg_solve
std::tuple<Tensor, Tensor> linalg_solve_ex(const Tensor& A,
                                           const Tensor& B,
                                           bool left,
                                           bool check_errors) {
  Tensor result, LU, pivots, info;
  std::tie(result, LU, pivots, info) = at::_linalg_solve_ex(A, B, left, check_errors);
  return std::make_tuple(std::move(result), std::move(info));
}

Tensor& linalg_solve_out(const Tensor& A,
                         const Tensor& B,
                         bool left,
                         Tensor& result) {
  auto info = B.new_empty({0}, kInt);
  at::linalg_solve_ex_out(result, info, A, B, left);
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
  return result;
}

Tensor linalg_solve(const Tensor& A,
                    const Tensor& B,
                    bool left) {
  Tensor result, info;
  std::tie(result, info) = at::linalg_solve_ex(A, B, left);
  at::_linalg_check_errors(info, "torch.linalg.solve", A.dim() == 2);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_factor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(lu_factor_stub);

TORCH_IMPL_FUNC(linalg_lu_factor_ex_out)(const Tensor& A,
                                         bool pivot,
                                         bool check_errors,
                                         const Tensor& LU,
                                         const Tensor& pivots,
                                         const Tensor& info) {
  if (A.numel() == 0) {
    // zero out the infos as it will have one element if the input is a matrix of size (0, 0)
    info.zero_();
    return;
  }
  if (!LU.is_same(A)) {
    LU.copy_(A);
  }

  lu_factor_stub(A.device().type(), LU, pivots, info, pivot);

  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", A.dim() == 2);
  }
}

std::tuple<Tensor&, Tensor&> linalg_lu_factor_out(const Tensor& A, bool pivot, Tensor& LU, Tensor& pivots) {
  auto info = at::empty({0}, A.options().dtype(kInt));
  // We pass check_errors as we want to use lu_factor rather than lu_factor_ex in the errors
  at::linalg_lu_factor_ex_out(LU, pivots, info, A, pivot, /*check_errors=*/false);
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  return std::tie(LU, pivots);
}

std::tuple<Tensor, Tensor> linalg_lu_factor(const Tensor& A, bool pivot) {
  Tensor LU, pivots, info;
  std::tie(LU, pivots, info) = at::linalg_lu_factor_ex(A, pivot, /*check_errors=*/false);
  at::_linalg_check_errors(info, "torch.linalg.lu_factor", A.dim() == 2);
  return std::make_tuple(std::move(LU), std::move(pivots));
}

// TODO Deprecate this function in favour of linalg_lu_factor_ex
std::tuple<Tensor, Tensor, Tensor> _lu_with_info(const Tensor& self, bool compute_pivots, bool) {
   TORCH_WARN_ONCE(
    "torch.lu is deprecated in favor of torch.linalg.lu_factor / torch.linalg.lu_factor_ex and will be ",
    "removed in a future PyTorch release.\n",
    "LU, pivots = torch.lu(A, compute_pivots)\n",
    "should be replaced with\n",
    "LU, pivots = torch.linalg.lu_factor(A, compute_pivots)\n",
    "and\n",
    "LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)\n",
    "should be replaced with\n",
    "LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)"
  );
  return at::linalg_lu_factor_ex(self, compute_pivots, false);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(unpack_pivots_stub);

TORCH_IMPL_FUNC(linalg_lu_out)(const Tensor& A,
                               bool pivot,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  const auto m = A.sizes().end()[-2];
  const auto n = A.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // with k = min(m, n)

  // Use L as it has the correct size
  const bool use_L = m > n;
  auto pivots = at::empty({0}, A.options().dtype(kInt));
  auto info = at::empty({0}, A.options().dtype(kInt));
  at::linalg_lu_factor_ex_out(const_cast<Tensor&>(use_L ? L : U),
                              const_cast<Tensor&>(pivots),
                              const_cast<Tensor&>(info),
                              A,
                              pivot,
                              /*check_errors=*/false);
  at::lu_unpack_out(const_cast<Tensor&>(P),
                    const_cast<Tensor&>(L),
                    const_cast<Tensor&>(U),
                    use_L ? L : U,
                    pivots,
                    /*unpack_lu=*/true,
                    /*unpack_pivots=*/pivot);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_unpack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TORCH_IMPL_FUNC(lu_unpack_out)(const Tensor& LU,
                               const Tensor& pivots,
                               bool unpack_lu,
                               bool unpack_pivots,
                               const Tensor& P,
                               const Tensor& L,
                               const Tensor& U) {
  const auto m = LU.sizes().end()[-2];
  const auto n = LU.sizes().end()[-1];

  // A.shape[-2:] == (m, n)
  // P.shape[-2:] == (m, m)
  // L.shape[-2:] == (m, k)
  // U.shape[-2:] == (k, n)
  // with k = min(m, n)

  if (unpack_lu) {
    if (m > n || LU.is_same(L)) {
      // The order of triu and tril is important as we may have LU.is_same(L)
      at::triu_out(const_cast<Tensor&>(U), m == n ? LU : LU.narrow(-2, 0, n), 0);
      at::tril_out(const_cast<Tensor&>(L), LU, -1);
      L.diagonal(0, -2, -1).fill_(1.);
    } else {
      // The order of triu and tril is important as we may have LU.is_same(U)
      at::tril_out(const_cast<Tensor&>(L), m == n ? LU : LU.narrow(-1, 0, m), -1);
      L.diagonal(0, -2, -1).fill_(1.);
      at::triu_out(const_cast<Tensor&>(U), LU, 0);
    }
  }
  if (unpack_pivots) {
    // lu_factor_ex returns an int32 1-based indexing, which is what we have in `pivots`
    // We transform that to a proper permutation of the indices {0, ..., m-1}
    const auto perm_sizes = IntArrayRef(P.sizes().data(), P.dim() - 1);

    // Fill `perm` with the identity permutation (perhaps batched)
    const auto perm = at::arange(m, pivots.options().memory_format(at::MemoryFormat::Contiguous).dtype(kLong))
                        .expand(perm_sizes)
                        .contiguous();

    // Note that perm is of type kLong and pivots is a 1-indexed kInt.
    // This is taken into account in the unpack_pivots kernel
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .declare_static_shape(pivots.sizes(), /*squash_dim=*/pivots.dim() - 1)
      .add_output(perm)
      .add_owned_input(pivots.contiguous())
      .build();

    unpack_pivots_stub(pivots.device().type(), iter, std::min(m, n), m);

    // Transform the permutation into a permutation matrix
    P.zero_();
    P.scatter_(-2, perm.unsqueeze(-2), 1.);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DEFINE_DISPATCH(lu_solve_stub);

TORCH_IMPL_FUNC(linalg_lu_solve_out)(const Tensor& LU,
                                     const Tensor& pivots,
                                     const Tensor& B,
                                     bool left,
                                     bool adjoint,
                                     const Tensor& result) {
  // Trivial case
  if (result.numel() == 0) {
    return;
  }

  // Solve A^H X = B^H. Then we return X^H
  if (!left) {
    adjoint = !adjoint;
    result.transpose_(-2, -1);
  }

  // Copy B (or B^H) into result
  if (!result.is_same(B)) {
    result.copy_(left ? B : B.mH());
  }

  // Make LU / pivots F-contiguous
  auto pivots_ = pivots.expect_contiguous();
  auto LU_ = at::native::borrow_else_clone(
      LU.mT().is_contiguous(), LU, LU, /*row_major=*/false);

  const auto trans = !adjoint ? TransposeType::NoTranspose :
                     LU.is_complex() ? TransposeType::ConjTranspose
                                     : TransposeType::Transpose;

  lu_solve_stub(LU_->device().type(), *LU_, *pivots_, result, trans);

  // Conj-transpose back in-place
  if (!left) {
    result.transpose_(-2, -1);
    if (result.is_complex()) {
      result._set_conj(!result.is_conj());
    }
  }
}

Tensor lu_solve(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots) {
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  return at::linalg_lu_solve(LU_data, LU_pivots, self);
}

Tensor& lu_solve_out(const Tensor& self, const Tensor& LU_data, const Tensor& LU_pivots, Tensor& result) {
  TORCH_WARN_ONCE(
    "torch.lu_solve is deprecated in favor of torch.linalg.lu_solve",
    "and will be removed in a future PyTorch release.\n",
    "Note that torch.linalg.lu_solve has its arguments reversed.\n",
    "X = torch.lu_solve(B, LU, pivots)\n",
    "should be replaced with\n",
    "X = torch.linalg.lu_solve(LU, pivots, B)"
  );
  return at::linalg_lu_solve_out(result, LU_data, LU_pivots, self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(triangular_solve_stub);

/*
Solves the matrix equation 'input' @ 'result' = 'other' for the 'result'.
The result of the computation is saved in-place in 'result' tensor,
'clone_input' will be a copy of 'input',
'infos' is used to store information for possible checks for error,
'upper' controls the portion of input matrix to consider in computations,
'transpose' if true then 'input.mT()' @ 'result' = 'other' is solved,
'unitriangular' if true then the diagonal elements of 'input' are assumed to be 1
and the actual diagonal values are not used.
*/
static void triangular_solve_out_impl(
    const Tensor& result,
    const Tensor& clone_input,
    const Tensor& input,
    const Tensor& other,
    bool upper, bool transpose, bool unitriangular) {
  TORCH_WARN_ONCE(
    "torch.triangular_solve is deprecated in favor of torch.linalg.solve_triangular",
    "and will be removed in a future PyTorch release.\n",
    "torch.linalg.solve_triangular has its arguments reversed and does not return a copy of one of the inputs.\n",
    "X = torch.triangular_solve(B, A).solution\n",
    "should be replaced with\n",
    "X = torch.linalg.solve_triangular(A, B).");
  // These internal asserts make explicit the assumptions in the implementation
  // Error check with the actual error messages are done on the higher level of
  // the hierarchy of calls
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.size(-2) == input.size(-1));

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == other.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == result.device());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == clone_input.device());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == other.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == result.scalar_type());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.scalar_type() == clone_input.scalar_type());

  // if 'result' has no elements we can modify it
  if (result.numel() == 0) {
    result.resize_(other.mT().sizes(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);  // make 'result' to have Fortran contiguous memory layout
  }

  // if 'clone_input' has no elements we can modify it
  if (clone_input.numel() == 0) {
    clone_input.resize_(input.mT().sizes(), MemoryFormat::Contiguous);
    clone_input.transpose_(-2, -1);  // make 'clone_input' to have Fortran contiguous memory layout
  }

  // 'result' and 'clone_input' must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.mT().is_contiguous());

  // triangular_solve_stub performs calculations in-place
  // 'result' must be a copy of 'other'
  // 'clone_input' must be a copy of 'input'
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(clone_input.sizes().equals(input.sizes()));
  result.copy_(other);
  clone_input.copy_(input);

  triangular_solve_stub(input.device().type(), clone_input, result, /*left=*/true, upper, transpose ? TransposeType::Transpose : TransposeType::NoTranspose, unitriangular);
}

TORCH_IMPL_FUNC(triangular_solve_out)(const Tensor& self, const Tensor& A, bool upper, bool transpose, bool unitriangular, const Tensor& result, const Tensor& clone_A) {
  Tensor self_broadcast, A_broadcast;
  std::tie(self_broadcast, A_broadcast) = _linalg_broadcast_batch_dims(self, A, "triangular_solve");

  bool copy_needed = !result.transpose(-2, -1).is_contiguous();
  copy_needed |= !clone_A.transpose(-2, -1).is_contiguous();

  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, self.options());
    Tensor clone_A_tmp = at::empty({0}, A.options());

    triangular_solve_out_impl(result_tmp, clone_A_tmp, A_broadcast, self_broadcast, upper, transpose, unitriangular);

    result.copy_(result_tmp);
    clone_A.copy_(clone_A_tmp);
  } else {
    triangular_solve_out_impl(result, clone_A, A_broadcast, self_broadcast, upper, transpose, unitriangular);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(geqrf_stub);

static void geqrf_out_helper(const Tensor& input, const Tensor& QR, const Tensor& tau) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);

  TORCH_INTERNAL_ASSERT(input.scalar_type() == QR.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == QR.device());

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  // if 'QR' has no elements we can modify it
  if (QR.numel() == 0) {
    QR.resize_as_(input.mT(), MemoryFormat::Contiguous);
    QR.transpose_(-2, -1); // make Fortran-contiguous
  }

  auto expected_batch_tau_shape = IntArrayRef(input.sizes().data(), input.dim() - 2).vec(); // input.shape[:-2]
  expected_batch_tau_shape.push_back(std::min(input.size(-2), input.size(-1)));
  if (tau.numel() == 0) {
    tau.resize_(expected_batch_tau_shape);
  }

  // QR tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(QR.mT().is_contiguous());
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
    is_batched_column_major = QR.mT().is_contiguous();
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
  return std::make_tuple(std::move(QR), std::move(tau));
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
TORCH_IMPL_FUNC(linalg_qr_out)(const Tensor& A,
                               c10::string_view mode,
                               const Tensor & Q,
                               const Tensor & R) {
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto k = std::min(m, n);
  bool compute_q, reduced_mode;
  std::tie(compute_q, reduced_mode) = at::native::_parse_qr_mode(mode);


  // We need an auxiliary tensor to call geqrf
  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = k;
  auto tau = A.new_empty(tau_shape);

  // geqrf requires m x n workspace input that is modified in-place
  // We try to use Q. If it doesn't fit, we try to use R
  // If m > n and compute_q==false, it won't fit into Q or R, so we neet to create an auxiliary tensor
  Tensor QR;
  if (compute_q && Q.size(-1) == n) {
    QR = Q;
    QR.copy_(A);
  } else if (R.size(-2) == m) {
    QR = R;
    QR.copy_(A);
  } else {
    QR = cloneBatchedColumnMajor(A);
  }

  geqrf_stub(A.device().type(), QR, tau);

  // Split QR into Q (unless compute_q == false) and R
  if (QR.is_alias_of(R)) {
    // Copy QR into Q
    if (compute_q) {
      // If the result didn't fit in Q and compute_q == true is because Q is not of size m x n (i.e. it's of size m x m)
      TORCH_INTERNAL_ASSERT(Q.size(-1) == m);
      if (m < n) {
        Q.copy_(QR.slice(-1, 0, m));
      } else {
        Q.slice(-1, 0, n).copy_(QR);
      }
    }
    R.triu_();
  } else {
    // Copy QR into R from Q or the aux tensor
    at::triu_out(const_cast<Tensor&>(R), QR.slice(-2, 0, n));
  }

  if (compute_q) {
    // Next perform ORGQR for Q using the result from GEQRF
    orgqr_stub(A.device().type(), const_cast<Tensor&>(Q), tau);
  }
}


std::tuple<Tensor,Tensor> qr(const Tensor& self, bool some) {
  TORCH_WARN_ONCE(
    "torch.qr is deprecated in favor of torch.linalg.qr and will be removed in a future PyTorch release.\n",
    "The boolean parameter 'some' has been replaced with a string parameter 'mode'.\n",
    "Q, R = torch.qr(A, some)\n",
    "should be replaced with\n",
    "Q, R = torch.linalg.qr(A, 'reduced' if some else 'complete')"
  );
  const char* mode = some ? "reduced" : "complete";
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
  const char* mode = some ? "reduced" : "complete";
  return at::linalg_qr_out(Q, R, self, mode);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
static Tensor& householder_product_out_helper(const Tensor& input, const Tensor& tau, Tensor& result) {
  TORCH_INTERNAL_ASSERT(input.dim() >= 2);
  TORCH_INTERNAL_ASSERT(input.size(-2) >= input.size(-1));
  TORCH_INTERNAL_ASSERT(input.size(-1) >= tau.size(-1));

  TORCH_INTERNAL_ASSERT(input.scalar_type() == tau.scalar_type());
  TORCH_INTERNAL_ASSERT(input.device() == tau.device());

  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    at::native::resize_as_(result, input.mT(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // result tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT(result.mT().is_contiguous());
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
  checkSameDevice("torch.linalg.householder_product", tau, input, "tau");
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
    is_batched_column_major = result.mT().is_contiguous();
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

DEFINE_DISPATCH(ormqr_stub);

static void ormqr_out_helper(const Tensor& input, const Tensor& tau, const Tensor& other, const Tensor& result, bool left, bool transpose) {
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
    at::native::resize_as_(result, other.mT(), MemoryFormat::Contiguous);
    result.transpose_(-2, -1);
  }

  // 'result' tensor must be in batched column major order (Fortran contiguous)
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.sizes().equals(other.sizes()));

  // 'tau' tensor must be contiguous
  Tensor tau_ = tau;
  if (!tau.is_contiguous()) {
    tau_ = at::empty(tau.sizes(), tau.options(), MemoryFormat::Contiguous);
    tau_.copy_(tau);
  }

  // 'input' tensor must be Fortran contiguous
  Tensor input_ = input;
  if (!input.mT().is_contiguous()) {
    input_ = at::empty(input.mT().sizes(), input.options(), MemoryFormat::Contiguous);
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
      tau.size(-1) <= input.size(-1),
      "torch.ormqr: tau.shape[-1] must be less than or equal to input.shape[-1]");

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
    is_batched_column_major = result.mT().is_contiguous();
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

DEFINE_DISPATCH(linalg_eigh_stub);

/*
  Computes eigenvalues and eigenvectors of the tensor 'input'.

  Args:
  * 'input' - input Tensor for eigendecomposition
  * 'values' - Tensor to store computed eigenvalues
  * 'vectors' - Tensor to store computed eigenvectors
  * 'infos' - Tensor to store LAPACK/MAGMA/cuSOLVER error codes
  * 'compute_eigenvectors' - controls whether eigenvectors should be computed
  * 'uplo' - controls the portion of input matrix to consider in computations, allowed values are "u", "U", "l", "L"
    "u", "U" - upper triangular portion of the input matrix is used in computations; "l", "L" - lower.
*/

TORCH_IMPL_FUNC(_linalg_eigh_out)(const Tensor& A,
                                  c10::string_view uplo,
                                  bool compute_v,
                                  const Tensor& L,
                                  const Tensor& V) {
  if (A.numel() == 0) {
    return;
  }

  auto uplo_uppercase = static_cast<char>(std::toupper(static_cast<unsigned char>(uplo[0])));
  bool upper = (uplo_uppercase == 'U');

  Tensor V_ = V;
  if (compute_v) {
    V_.copy_(A);
  } else {
    // We need a tensor to hold A
    V_ = cloneBatchedColumnMajor(A);
  }

  const auto info = at::zeros(A.sizes().slice(0, A.dim() - 2), A.options().dtype(kInt));
  linalg_eigh_stub(A.device().type(), L, V_, info, upper, compute_v);

  at::_linalg_check_errors(info, "linalg.eigh", /*is_matrix*/A.dim() == 2);
}

std::tuple<Tensor, Tensor> linalg_eigh(const Tensor& A, c10::string_view uplo) {
  // TODO (Good intro task) Implement linalg_eigh_ex_out
  return at::_linalg_eigh(A, uplo, /*compute_v*/true);
}

std::tuple<Tensor&, Tensor&> linalg_eigh_out(const Tensor& A, c10::string_view uplo, Tensor& L, Tensor& V) {
  return at::_linalg_eigh_out(L, V, A, uplo, /*compute_v=*/true);
}


Tensor linalg_eigvalsh(const Tensor& A, c10::string_view uplo) {
  return std::get<0>(at::_linalg_eigh(A, uplo,
                     /*compute_v=*/_may_require_fw_or_bw_grad(A)));
}

Tensor& linalg_eigvalsh_out(const Tensor& A, c10::string_view uplo, Tensor& L) {
  auto V = at::empty({0}, A.options());
  at::_linalg_eigh_out(L, V, A, uplo, /*comptue_v=*/false);
  return L;
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

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.mT().is_contiguous());

  AT_DISPATCH_FLOATING_TYPES(real_vectors.scalar_type(), "linalg_eig_make_complex_vector", [&]{
    linalg_eig_make_complex_eigenvectors_impl<scalar_t>(complex_vectors, complex_values, real_vectors);
  });
  return complex_vectors;
}

DEFINE_DISPATCH(linalg_eig_stub);

static std::tuple<Tensor&, Tensor&> linalg_eig_out_info(const Tensor& input, Tensor& values, Tensor& vectors, Tensor& infos, bool compute_eigenvectors) {
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
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.mT().is_contiguous());
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
  TORCH_CHECK(input.isfinite().all().item<bool>(), "torch.linalg.eig: input tensor should not contain infs or NaNs.");
  squareCheckInputs(input, "linalg.eig");

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
    is_batched_column_major = vectors.mT().is_contiguous();
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
  at::_linalg_check_errors(infos, "torch.linalg.eig", input.dim() == 2);
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
  squareCheckInputs(input, "linalg.eigvals");

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
  at::_linalg_check_errors(infos, "torch.linalg.eigvals", input.dim() == 2);
  return values;
}

Tensor linalg_eigvals(const Tensor& input) {
  // if input requires grad we must compute the eigenvectors to make this function differentiable
  // the eigenvectors are not exposed to the user
  if (_may_require_fw_or_bw_grad(input)) {
    return std::get<0>(at::linalg_eig(input));
  }

  ScalarType complex_dtype = toComplexType(input.scalar_type());
  Tensor values = at::empty({0}, input.options().dtype(complex_dtype));

  at::linalg_eigvals_outf(input, values);

  return values;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/* torch.svd, implemented in terms of torch.linalg.svd. There are two main
   differences:

    1. the 2nd parameter is bool some=True, which if effectively the opposite
       of full_matrices=True

    2. svd returns V, while linalg.svd returns Vh = V^H
*/

DEFINE_DISPATCH(svd_stub);

TORCH_IMPL_FUNC(_linalg_svd_out)(const Tensor& A,
                                 const bool full_matrices,
                                 const bool compute_uv,
                                 c10::optional<c10::string_view> driver,
                                 const Tensor & U,
                                 const Tensor & S,
                                 const Tensor & Vh) {
  // Half optimisation half precondition for some parts of the LAPACK / cuSOLVER
  // In particular, the call to lapackSvd to compute lwork fails otherwise
  if (A.numel() == 0) {
    // Needed in the case that we have e.g. A.shape == (3, 0) and full_matrices=True
    // We fill U or Vh with the identity matrix as it's a valid SVD for the empty matrix
    if (compute_uv && full_matrices) {
      if (U.numel() != 0) {
        U.zero_();
        U.diagonal(0, -2, -1).fill_(1.);
      }
      if (Vh.numel() != 0) {
        Vh.zero_();
        Vh.diagonal(0, -2, -1).fill_(1.);
      }
    }
    return;
  }

  // We need to distinguish the cuSOLVER case, as cuSOLVER expects F-contig matrices, but
  // it computes V rather than Vh
  const bool use_cusolver = at::native::svd_uses_cusolver(A);
  TORCH_CHECK(use_cusolver || !driver.has_value(),
    "torch.linalg.svd: keyword argument `driver=` is only supported on CUDA inputs with cuSOLVER backend.");

  // A always needs to be copied as its contents will be destroyed during the computaton of the SVD
  // Now, MAGMA needs the copy to be on CPU, while cuSOLVER needs it to be on CUDA, so we'll defer
  // the copy as a column major matrix to the backends.
  const auto info = at::zeros(IntArrayRef(A.sizes().begin(), A.sizes().end() - 2), A.options().dtype(kInt));

  svd_stub(A.device().type(),
           A,
           full_matrices,
           compute_uv,
           driver,
           U, S, Vh, info);

  // TODO This should be removed, and the code checking for convergence should be lifted
  // from svd_cusolver to this function. We should then make sure that this function
  // never errors out.
  at::_linalg_check_errors(info, "linalg.svd", /*is_matrix*/A.dim() == 2);
}

std::tuple<Tensor&, Tensor&, Tensor&>
linalg_svd_out(const Tensor& A,
               bool full_matrices,
               c10::optional<c10::string_view> driver,
               Tensor & U,
               Tensor & S,
               Tensor & Vh) {
  // This function does not have an _ex variant as we always check errors inside
  // to assure the convergence of the algorithm anyway. See
  // https://github.com/pytorch/pytorch/issues/28293
  // https://github.com/pytorch/pytorch/issues/64237
  //
  // We must delegate both linalg_svd and linalg_svdvals to
  // _linalg_svd (rather than delegating linalg_svdvals to linalg_svd) because
  //   1. We don't want to expose the `compute_uv` parameter in svd
  //   2. We would like to make use of the `compute_uv=False` optimisation within svdvals
  // The only way to achieve these two things and still abide by the compositionality rules
  // is by dispatching to another function.
  return at::_linalg_svd_out(U, S, Vh, A, full_matrices, /*compute_uv=*/true, driver);
}

std::tuple<Tensor, Tensor, Tensor> linalg_svd(const Tensor& A, bool full_matrices,
    c10::optional<c10::string_view> driver) {
  return at::_linalg_svd(A, full_matrices, /*compute_uv=*/true, driver);
}

// See note in linalg_svd for why this function does not have an _ex variant
Tensor& linalg_svdvals_out(const Tensor& A, c10::optional<c10::string_view> driver, Tensor & S) {
  // Dummies
  auto U = at::empty({0}, A.options());
  auto Vh = at::empty({0}, A.options());
  at::_linalg_svd_out(U, S, Vh, A, /*full_matrices=*/false, /*comptue_uv=*/false, /*driver=*/driver);
  return S;
}

Tensor linalg_svdvals(const Tensor& A, c10::optional<c10::string_view> driver) {
  return std::get<1>(at::_linalg_svd(A, /*full_matrices=*/false,
                     /*compute_uv=*/_may_require_fw_or_bw_grad(A),
                     /*driver=*/driver));
}

std::tuple<Tensor&, Tensor&, Tensor&> svd_out(const Tensor& self, bool some, bool compute_uv,
    Tensor& U, Tensor& S, Tensor& V) {

  if (compute_uv) {
    if (V.dim() >= 2) {
      V.transpose_(-2, -1);
    }
    at::linalg_svd_out(U, S, V, self, /*full_matrices=*/!some);
    V.transpose_(-2, -1);
    if (V.is_complex()) {
      // We cannot use `_set_conj` as it does not play well with backwards
      V.conj_physical_();
    }
  } else {
    TORCH_CHECK(self.scalar_type() == U.scalar_type(),
    "torch.svd: Expected out tensor to have dtype ", self.scalar_type(), " but got ", U.scalar_type(), " instead");

    TORCH_CHECK(self.scalar_type() == V.scalar_type(),
    "torch.svd: Expected out tensor to have dtype ", self.scalar_type(), " but got ", V.scalar_type(), " instead");

    at::linalg_svdvals_out(S, self);
    // some == false returns U, Vh of size (m, m), (n, n) full of zeros
    const auto m = self.size(-2);
    const auto n = self.size(-1);
    auto sizes = self.sizes().vec();

    sizes.end()[-1] = m;
    at::native::resize_output(U, sizes);
    U.zero_();

    sizes.end()[-2] = n;
    sizes.end()[-1] = n;
    at::native::resize_output(V, sizes);
    V.zero_();
  }

  return std::tie(U, S, V);
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
  //     "V = Vh.mH()\n",
  //     "and\n",
  //     "_, S, _ = torch.svd(A, some=some, compute_uv=False)\n",
  //     "should be replaced with\n",
  //     "S = torch.linalg.svdvals(A)");
  TORCH_CHECK(self.dim() >= 2, "linalg.svd: input should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  Tensor U, S, Vh;
  if (compute_uv) {
    std::tie(U, S, Vh) = at::linalg_svd(self, /*full_matrices=*/!some);
  } else {
    S = at::linalg_svdvals(self);
    // some == false returns U, Vh of size (m, m), (n, n) full of zeros
    const auto m = self.size(-2);
    const auto n = self.size(-1);

    auto sizes = self.sizes().vec();
    sizes.end()[-1] = m;
    U = at::zeros(sizes, self.options());
    sizes.end()[-2] = n;
    sizes.end()[-1] = n;
    Vh = at::zeros(sizes, self.options());
  }
  return std::make_tuple(std::move(U), std::move(S), Vh.mH());
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

  auto real_dtype = toRealValueType(input.scalar_type());
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
    TORCH_INTERNAL_ASSERT(solution.mT().is_contiguous());
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
  auto solution_view = solution.narrow(/*dim=*/-2, /*start=*/0, /*length*/n);
  // manually restride original
  solution.set_(solution.storage(), solution_view.storage_offset(), solution_view.sizes(), solution_view.strides());
  if (m == 0) {
    solution.zero_();
  }

  // for 1-dimensional 'other', we need to squeeze the solution after "apply_lstsq"
  if (vector_case) {
    solution.squeeze_(-1);
  }
}

static std::string get_default_lstsq_driver(c10::optional<c10::string_view> driver, const Tensor& input) {
  // if `driver` is empty, we set driver_str to "gels" if working with CUDA tensors,
  // otherwise to "gelsy" driver.
  std::string driver_str;
  // check whether the user provided name is a valid driver name
  if (driver.has_value()) {
    driver_str = std::string(driver.value());
    // convert `driver_str` to lower case inplace.
    std::transform(driver_str.begin(), driver_str.end(), driver_str.begin(),
      [](unsigned char c) { return std::tolower(c); });
    static std::unordered_set<c10::string_view> allowed_drivers = {
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
    c10::optional<c10::string_view> driver,
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
  ScalarType real_dtype = c10::toRealValueType(input.scalar_type());
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
    : _get_epsilon(c10::toRealValueType(input.scalar_type())) * std::max<int64_t>(input.size(-2), input.size(-1));

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
    is_solution_batched_column_major = solution.mT().is_contiguous();
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

  at::_linalg_check_errors(infos, "torch.linalg.lstsq", infos.numel() <= 1);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(solution, residuals, rank, singular_values);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq(
    const Tensor& input, const Tensor& other,
    c10::optional<double> rcond,
    c10::optional<c10::string_view> driver) {
  Tensor solution = at::empty({0}, input.options());
  Tensor residuals = at::empty({0}, input.options().dtype(toRealValueType(input.scalar_type())));
  Tensor rank = at::empty({0}, input.options().dtype(at::kLong));
  Tensor singular_values = at::empty({0}, input.options().dtype(toRealValueType(input.scalar_type())));
  std::tie(solution, residuals, rank, singular_values) =
      at::linalg_lstsq_outf(input, other, rcond, driver, solution, residuals, rank, singular_values);
  return std::make_tuple(std::move(solution), std::move(residuals), std::move(rank), std::move(singular_values));
}

DEFINE_DISPATCH(ldl_factor_stub);

TORCH_IMPL_FUNC(linalg_ldl_factor_ex_out)
(const Tensor& self,
 bool hermitian,
 bool check_errors,
 const Tensor& LD,
 const Tensor& pivots,
 const Tensor& info) {
  // LAPACK workspace query segfalts if the input has 0 in batch dimensions.
  if (self.numel() == 0) {
    info.zero_();
    return;
  }

  // We decided not to include upper flag in the API.
  // https://github.com/pytorch/pytorch/pull/69828#issuecomment-1015143819
  // We can revisit this decision later and remove upper completely
  // also from low level functions or add it to the public API.
  bool upper = false;
  if (upper) {
    at::triu_out(const_cast<Tensor&>(LD), self);
  } else {
    at::tril_out(const_cast<Tensor&>(LD), self);
  }

  // call ldl_factor_stub that fills the result tensors
  ldl_factor_stub(
      self.device().type(), LD, pivots, info, upper, hermitian);

  if (check_errors) {
    at::_linalg_check_errors(
        info, "torch.linalg.ldl_factor_ex", self.dim() == 2);
  }
}

std::tuple<Tensor&, Tensor&> linalg_ldl_factor_out(
    const Tensor& self,
    bool hermitian,
    Tensor& LD,
    Tensor& pivots) {
  auto info = at::empty({0}, self.options().dtype(kInt));
  // We pass check_errors as we want to use lu_factor rather than lu_factor_ex
  // in the errors
  at::linalg_ldl_factor_ex_outf(
      self, hermitian, /*check_errors=*/false, LD, pivots, info);
  at::_linalg_check_errors(info, "torch.linalg.ldl_factor", self.dim() == 2);
  return std::tie(LD, pivots);
}

std::tuple<Tensor, Tensor> linalg_ldl_factor(
    const Tensor& self,
    bool hermitian) {
  Tensor LD, pivots, info;
  std::tie(LD, pivots, info) =
      at::linalg_ldl_factor_ex(self, hermitian, /*check_errors=*/false);
  at::_linalg_check_errors(info, "torch.linalg.ldl_factor", self.dim() == 2);
  return std::make_tuple(std::move(LD), std::move(pivots));
}

DEFINE_DISPATCH(ldl_solve_stub);

TORCH_IMPL_FUNC(linalg_ldl_solve_out)
(const Tensor& LD,
 const Tensor& pivots,
 const Tensor& B,
 bool hermitian,
 const Tensor& result) {
  if (LD.numel() == 0 || pivots.numel() == 0) {
    return;
  }

  auto pivots_ = pivots.expect_contiguous();

  auto LD_ = at::native::borrow_else_clone(
      LD.mT().is_contiguous(), LD, LD, /*row_major=*/false);
  result.copy_(B);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(result) == batchCount(result));

  ldl_solve_stub(
      B.device().type(), *LD_, *pivots_, result, false, hermitian);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ solve_triangular ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& linalg_vecdot_out(const Tensor& x, const Tensor& y, int64_t dim, Tensor& out) {
  checkFloatingOrComplex(x, "linalg.vecdot");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(),
              "linalg.vecdot: Expected x and y to have the same dtype, but found x of type ",
              x.scalar_type(), " and y of type ", y.scalar_type(), " instead");
  // out checks
  TORCH_CHECK(out.scalar_type() == x.scalar_type(),
              "linalg.vecdot: Expected out of dtype", x.scalar_type(),
              " but found ", out.scalar_type());
  checkSameDevice("linalg.vecdot", x, out);

  // Computes x^H y
  if (x.dim() == 1 && y.dim() == 1) {
    at::native::resize_output(out, {});
    return at::vdot_out(out, x, y);
  } else {
    return at::sum_out(out, x.conj() * y, /*dim=*/dim);
  }
}

Tensor linalg_vecdot(const Tensor& x, const Tensor& y, int64_t dim) {
  checkFloatingOrComplex(x, "linalg.vecdot");
  TORCH_CHECK(x.scalar_type() == y.scalar_type(),
              "linalg.vecdot: Expected x and y to have the same dtype, but found x of type ",
              x.scalar_type(), " and y of type ", y.scalar_type(), " instead");
  // Computes x^H y
  if (x.dim() == 1 && y.dim() == 1) {
    return at::vdot(x, y);
  } else {
    return x.conj().mul(y).sum(/*dim=*/dim);
  }
}

/*
Solves the matrix equation AX = B for A triangular.
'left' If true solves AX = B, if false solves XA = B
'upper' controls the portion of input matrix to consider in computations,
'unitriangular' if true then we assume diag(A) to be ones
'out' The tensor with the result. If A == out, A will be modified in place
*/
Tensor& linalg_solve_triangular_out(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool left,
    bool unitriangular,
    Tensor& out) {
  checkInputsSolver(A, B, left, "linalg.solve_triangular");
  Tensor A_, B_;
  std::tie(B_, A_) = _linalg_broadcast_batch_dims(B, A, /*don't check errors*/nullptr);

  // We'll write F-contig / F-transpose for FORTRAN contiguous / FORTRAN transpose etc
  // We say that a matrix is F-ready if it's F-contig OR F-transpose
  // At this point, A, B have been broadcasted but may or may not be F-ready

  // The following algorithm minimises copies and allocations. In pseudocode:
  // if out is wrong size:
  //   resize_output(out)
  // # Invariant: out is the right size
  // Tensor out_f; # Tensor that we will pass to FORTRAN
  // if out is F-ready:
  //   out_f = out;
  // else:
  //   Allocate out_f F-ready
  // if B != out_f:
  //   copy B into out_f
  // # Invariant: out_f F-ready and has B copied into it
  // if out_f is F-transposed:
  //   transpose equation
  // if out_f is conj:
  //   conjugate equation
  // # Invariant: out_f is not conjugated and F-contig
  // Tensor A_f; # Tensor that will be sent to FORTRAN
  // if A is F-ready:
  //   if A is conj and A is not transposed:
  //     # We need to clone A in this case. See [Cloning A]
  //     clone A F-contig into A_f
  //   else:
  //     A_f = A;
  // else:
  //   clone A F-contig into A_f
  // # Invariant: out_f is F-contig and A_f is F-ready
  // # We pass FORTRAN the flags indicating if A_f is transposed and or conjugated
  //
  // # Here we undo the conjugations / transposes on out_f if needed
  //
  // if out_f not same out:
  //   copy out_f into out
  // return out
  //
  // Note: The logic for the negative bit is the same as that for the conjugate bit
  //
  // Note: [Cloning A] If we are careful when allocating B when it needs to be allocated at the
  // beginning of the algorithm, it is possible to always elide the copy of A here.
  // Via this trick, the algorithm will copy at most one of A or B (never both) whenever A
  // and B are F-ready and not A.is_neg() (which happens almost always in practice).
  // When called as f(A, B, out=B) in most practical cases it'll perform no copies.

  const bool avoid_copy_A = A_.transpose(-2, -1).is_contiguous() && A_.is_conj();
  if (avoid_copy_A) {
    // See Note: [Cloning A]
    at::native::resize_output(out, B_.sizes());
  }
  else {
    // poorman's reimplementation of resize_output with result F-contig
    if (resize_output_check(out, B_.sizes())) {
      out.resize_(B_.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
      out.transpose_(-2, -1);  // make 'out' have Fortran contiguous memory layout
    }
  }
  // Invariant: out has the right size, so we'll be able to copy into it later on

  Tensor out_f; // the out that will go into fortran
  // We use C10_LIKELY mostly for documentation as it helps following what's the most likely path
  if C10_LIKELY (is_row_or_column_contiguous(out)) {
    out_f = out;
    if C10_LIKELY (!out.is_same(B_)) {
      out_f.copy_(B_);
    }
  } else {
    if (avoid_copy_A) {
      // See Note: [Cloning A]
      out_f = B_.clone(at::MemoryFormat::Contiguous);
    }
    else {
      out_f = cloneBatchedColumnMajor(B_);
    }
  }
  // Invariant: out_f F-ready and has B copied into it

  // out_f is F-transposed
  bool transpose_A = false;
  bool transpose_out_f = false;
  if (out_f.stride(-1) == 1) {
    left = !left;
    transpose_A = true;
    transpose_out_f = true;
    out_f.transpose_(-2 ,-1);
  }

  // No need to conjugate anything if out_f is conj as AX = conj(B) <=> conj(A)conj(X) = B
  // and X = B after the algortihm. We just anotate that A is conjugated later on
  // The solution will be written into out_f, so it'll be conjugated already

  Tensor A_f = std::move(A_);  // The A that will go into fortran

  bool A_is_conj = A_f.is_conj() != out_f.is_conj();
  bool A_is_neg = A_f.is_neg() != out_f.is_neg();
  bool A_is_f_contig = (A_f.stride(-1) == 1) == transpose_A;
  if C10_UNLIKELY (!is_row_or_column_contiguous(A_f)) {
    // We first anotate with flags on A_f all the conj / transpose / neg coming from out
    // and then we clone the resulting tensor to resolve all of them in memory
    if (out_f.is_conj()) {
      A_f = A_f.conj();
    }
    A_is_conj = false;

    if (out_f.is_neg()) {
      A_f = A_f._neg_view();
    }
    A_is_neg = false;

    // This choice is to be consistent with how we flip `upper` later on
    // Note that this is the same reasoning we apply for neg and conj below
    // If B has neg or out or transpose, then we need to resolve it in memory
    A_f = transpose_A ? A_f.clone(at::MemoryFormat::Contiguous)
                      : cloneBatchedColumnMajor(A_f);
    A_is_f_contig = true;
  } else if C10_UNLIKELY (A_is_f_contig && A_is_conj) {
    if C10_UNLIKELY (A_f.is_neg() || out_f.is_neg()) {
      // Cases A_is_neg (remember that B.is_neg() iff out_f.is_same(B))
      // -AX = -B => A(-X) = B. Swap neg of A_f. Nothing to do on X as X.is_same(B).
      // -AX = B. We resolve the neg in memory
      // AX = -B => -A -X = B. We resolve the neg in memory for A,
      //                       Since X.is_same(B), we already have that X.is_neg() == true

      // We do the neg with a view, as this will be resolved in the clone below
      if (out_f.is_neg()) {
        A_f = A_f._neg_view();
      }
      A_is_neg = false;
    }
    // We resolve the transpose if necessary and then leave A_f F-transposed,
    // as BLAS can handle the case F-transposed and conjugated
    A_f = at::clone(transpose_A ? A_f.mT() : A_f, at::MemoryFormat::Contiguous);
    A_is_f_contig = false;
    if (transpose_A) {
      upper = !upper;
    }
    // As we've already resolved the conj of A in the clone
    A_is_conj = out_f.is_conj();
  } else if C10_UNLIKELY (A_is_neg) {
    // We follow the same logic as above, only that in this case we need to perform the
    // negation in memory
    if (out_f.is_neg()) {
      A_f = -A_f;
    } else {
      A_f = A_f.resolve_neg();
    }
    A_is_neg = false;
    // As we've already resolved the conj of A in the negationa bove
    A_is_conj = out_f.is_conj();
  }
  // Invariant: out_f is F-contig and A_f is F-ready
  // neg has been resolved

  // If we pass the matrix physically F-transposed, we need to change the parity of upper
  if (A_f.stride(-1) == 1) {
    upper = !upper;
  }

  triangular_solve_stub(
    A_f.device().type(), A_f, out_f,
    /*left=*/left,
    /*upper=*/upper,
    /*transpose*/to_transpose_type(A_is_f_contig, A_is_conj),
    /*unitriangular=*/unitriangular);

  if (transpose_out_f) {
    out_f.transpose_(-2, -1);
  }

  if (!out_f.is_same(out)) {
    out.copy_(out_f);
  }
  return out;
}

Tensor linalg_solve_triangular(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool left,
    bool unitriangular) {
  Tensor out = at::empty({0}, A.options());
  linalg_solve_triangular_out(A, B, upper, left, unitriangular, out);
  return out;
}

Tensor linalg_vander_symint(
    const Tensor& x,
    c10::optional<c10::SymInt> N) {
  auto t = x.scalar_type();
  TORCH_CHECK(t == ScalarType::Float ||
              t == ScalarType::Double ||
              t == ScalarType::ComplexFloat ||
              t == ScalarType::ComplexDouble ||
              c10::isIntegralType(t, false),
              "linalg.vander supports floating point, complex, and integer tensors, but got ", t);
  const auto x_ = x.dim() == 0 ? x.unsqueeze(-1) : x;

  auto shape = x_.sym_sizes().vec();
  const auto n = N.value_or(shape.back());
  TORCH_CHECK(n > 1, "N must be greater than 1.");

  // Append cumprod of the oher 0...n-1 powers
  shape.push_back(n - 1);
  auto result = at::cumprod(x_.unsqueeze(-1).expand_symint(shape), -1);
  // The row of ones
  shape.back() = 1LL;
  auto ones =  result.new_ones_symint(shape);
  return at::cat({std::move(ones), std::move(result)}, /*dim=*/ -1);
}
}}  // namespace at::native
