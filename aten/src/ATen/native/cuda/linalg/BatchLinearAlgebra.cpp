#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <utility>

#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <c10/util/Exception.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>
#include <ATen/native/cuda/linalg/MagmaUtils.h>
#include <ATen/native/cpu/zmath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cholesky_solve_helper_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_linalg_check_errors.h>
#endif

#if AT_MAGMA_ENABLED()
#include <magma_types.h>
#include <magma_v2.h>
#include <ATen/cuda/detail/CUDAHooks.h>

const bool use_magma_ = true;

namespace {
struct MagmaInitializer {
  MagmaInitializer() {
#if defined(BUILD_LAZY_CUDA_LINALG)
    magma_init();
#else
    ::at::cuda::detail::set_magma_init_fn([]{ magma_init(); });
#endif
  }
} initializer;
}  // namespace (anonymous)

#define AT_MAGMA_VERSION MAGMA_VERSION_MAJOR*100 + MAGMA_VERSION_MINOR*10 + MAGMA_VERSION_MICRO

// Check that MAGMA never releases MAGMA_VERSION_MINOR >= 10 or MAGMA_VERSION_MICRO >= 10
#if MAGMA_VERSION_MINOR >= 10 || MAGMA_VERSION_MICRO >= 10
#error "MAGMA release minor or micro version >= 10, please correct AT_MAGMA_VERSION"
#endif

#else
const bool use_magma_ = false;

#endif

namespace at::native {
#if defined(BUILD_LAZY_CUDA_LINALG)
// All registrations with PyTorch runtime should be done dynamically
// so if library is lazy loaded it must not export anything, otherwise
// it can result in symbol clashes
namespace lazy_linalg {
#endif

#if AT_MAGMA_ENABLED()

template <class scalar_t>
void magmaLdlHermitian(
    magma_uplo_t uplo,
    magma_int_t n,
    scalar_t* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  TORCH_CHECK(
      false,
      "LDL decomposition is not available.",
      "Please rebuild with MAGMA 2.5.4+.");
}

template<class scalar_t>
void magmaLu(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info);

template<class scalar_t>
void magmaLuBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaLuNoPiv(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    magma_int_t* info);

template<class scalar_t>
void magmaLuNoPivBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaCholeskySolve(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info);

template<class scalar_t>
void magmaCholeskySolveBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue);

#if defined(USE_ROCM)
template<class scalar_t>
void magmaTriangularSolveBatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t** dA_array, magma_int_t ldda, scalar_t** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);
#endif // defined(USE_ROCM)

#if defined(USE_ROCM) || !(defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702))
template<class scalar_t, class value_t=scalar_t>
void magmaEig(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, scalar_t *A, magma_int_t lda,
    scalar_t *w, scalar_t *VL, magma_int_t ldvl,
    scalar_t *VR, magma_int_t ldvr, scalar_t *work, magma_int_t lwork,
    value_t *rwork,
    magma_int_t *info);
#endif

#if AT_MAGMA_VERSION >= 254

template <>
void magmaLdlHermitian<double>(
    magma_uplo_t uplo,
    magma_int_t n,
    double* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dsytrf_gpu(uplo, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<float>(
    magma_uplo_t uplo,
    magma_int_t n,
    float* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_ssytrf_gpu(uplo, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<c10::complex<double>>(
    magma_uplo_t uplo,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zhetrf_gpu(
      uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaLdlHermitian<c10::complex<float>>(
    magma_uplo_t uplo,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    magma_int_t* ipiv,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_chetrf_gpu(
      uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

#endif // AT_MAGMA_VERSION >= 254

template<>
void magmaLu<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_gpu(m, n, dA, ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLu<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* ipiv, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_zgetrf_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magma_cgetrf_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgetrf_nopiv_gpu(m, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPiv<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgetrf_nopiv_gpu(m, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_sgetrf_nopiv_batched(m, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<double>>(
    magma_int_t m, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zgetrf_nopiv_batched(m, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuNoPivBatched<c10::complex<float>>(
    magma_int_t m, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cgetrf_nopiv_batched(m, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda,
    double* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda,
    float* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrs_gpu(uplo, n, nrhs, dA, ldda, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
    reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolve<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrs_gpu(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex*>(dA), ldda,
    reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_zpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskySolveBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  info = magma_cpotrs_batched(uplo, n, nrhs,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

#if defined(USE_ROCM)
template<>
void magmaTriangularSolveBatched<double>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t ldda, double** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_dtrsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<float>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    float** dA_array, magma_int_t ldda, float** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmablas_strsm_batched(side, uplo, trans, diag, m, n, 1, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<double>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<double>** dA_array, magma_int_t ldda, c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaDoubleComplex alpha({1, 0});
  magmablas_ztrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda,
    reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaTriangularSolveBatched<c10::complex<float>>(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    c10::complex<float>** dA_array, magma_int_t ldda, c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  magmaFloatComplex alpha({1, 0});
  magmablas_ctrsm_batched(side, uplo, trans, diag, m, n, alpha,
    reinterpret_cast<magmaFloatComplex**>(dA_array), ldda,
    reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}
#endif // defined(USE_ROCM)

#if defined(USE_ROCM) || !(defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702))
template<>
void magmaEig<double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    double *w,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  // magma [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  double *wr = w;
  double *wi = w + n;
  (void)rwork; // unused
  magma_dgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    float *w,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  float *wr = w;
  float *wi = w + n;
  (void)rwork; // unused
  magma_sgeev(jobvl, jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr, work, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<c10::complex<double>, double>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<double> *A, magma_int_t lda,
    c10::complex<double> *w,
    c10::complex<double> *VL, magma_int_t ldvl,
    c10::complex<double> *VR, magma_int_t ldvr,
    c10::complex<double> *work, magma_int_t lwork,
    double *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_zgeev(jobvl, jobvr, n,
         reinterpret_cast<magmaDoubleComplex*>(A), lda,
         reinterpret_cast<magmaDoubleComplex*>(w),
         reinterpret_cast<magmaDoubleComplex*>(VL), ldvl,
         reinterpret_cast<magmaDoubleComplex*>(VR), ldvr,
         reinterpret_cast<magmaDoubleComplex*>(work), lwork,
         rwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaEig<c10::complex<float>, float>(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    c10::complex<float> *A, magma_int_t lda,
    c10::complex<float> *w,
    c10::complex<float> *VL, magma_int_t ldvl,
    c10::complex<float> *VR, magma_int_t ldvr,
    c10::complex<float> *work, magma_int_t lwork,
    float *rwork,
    magma_int_t *info) {
  MagmaStreamSyncGuard guard;
  magma_cgeev(jobvl, jobvr, n,
         reinterpret_cast<magmaFloatComplex*>(A), lda,
         reinterpret_cast<magmaFloatComplex*>(w),
         reinterpret_cast<magmaFloatComplex*>(VL), ldvl,
         reinterpret_cast<magmaFloatComplex*>(VR), ldvr,
         reinterpret_cast<magmaFloatComplex*>(work), lwork,
         rwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}
#endif

namespace {

/*
  MAGMA can return errors both as a return value and in the info argument.
  The return value and info should always be identical.
  In general, the meaning is as given in this table.
  Predefined error codes are large negative numbers. Using the symbolic
  constants below is preferred, but the numeric values can be found in
  include/magma_types.h.

  Info                       |  Description
  -----------                |  -----------
  info = 0 (MAGMA_SUCCESS)   |  Successful exit
  info < 0, but small        |  For info = -i, the i-th argument had an illegal value
  info > 0                   |  Function-specific error such as singular matrix
  MAGMA_ERR_DEVICE_ALLOC     |  Could not allocate GPU device memory
  MAGMA_ERR_HOST_ALLOC       |  Could not allocate CPU host memory
  MAGMA_ERR_ILLEGAL_VALUE    |  An argument had an illegal value (deprecated; instead it should return -i to say the i-th argument was bad)
  MAGMA_ERR_INVALID_PTR      |  Can't free pointer
  MAGMA_ERR_NOT_IMPLEMENTED  |  Function or option not implemented
  MAGMA_ERR_NOT_SUPPORTED    |  Function or option not supported on the current architecture
*/
void checkMagmaInternalError(magma_int_t info, const std::string& magma_function_name) {
  // if info > 0 the error is function-specific, do nothing in this case
  TORCH_CHECK(info >= 0,
      "MAGMA error: ",
      magma_strerror(info),
      ", info = ", info,
      ", when calling ", magma_function_name);
}

magma_trans_t to_magma(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return MagmaNoTrans;
    case TransposeType::Transpose: return MagmaTrans;
    case TransposeType::ConjTranspose: return MagmaConjTrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}
} // anonymous namespace
#endif // AT_MAGMA_ENABLED()

#define ALLOCATE_ARRAY(name, type, size) \
  auto storage_##name = pin_memory<type>(size); \
  name = static_cast<type*>(storage_##name.mutable_data());

namespace {

void _warn_once_magma_deprecation(const std::string& op_name, bool force_cusolver = true) {
  if (at::globalContext().linalgPreferredBackend() == at::LinalgBackend::Magma) {
    std::string warn_force_cusolver = force_cusolver
      ? " " + op_name + " will try dispatching to cuSOLVER instead. " +
        "If you see any error messages, please, file an issue on GitHub."
      : "";
    TORCH_WARN_ONCE(
      op_name, ": "
      "MAGMA, as a linear algebra backend, is deprecated and will be removed "
      "in future releases.",
      warn_force_cusolver
    );
  }
}

template <typename scalar_t>
void apply_ldl_factor_magma(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "torch.linalg.ldl_factor: MAGMA library not found in "
      "compilation. Please rebuild with MAGMA.");
#else
  auto batch_size = batchCount(A);
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t leading_dim = magma_int_cast(A.stride(-1), "A.stride(-1)");
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.mutable_data_ptr<scalar_t>();
  Tensor pivots_cpu =
      at::empty_like(pivots, pivots.options().device(kCPU).pinned_memory(true));
  auto pivots_data = pivots_cpu.mutable_data_ptr<magma_int_t>();
  Tensor info_cpu =
      at::empty_like(info, info.options().device(kCPU).pinned_memory(true));
  auto info_data = info_cpu.mutable_data_ptr<magma_int_t>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* a_working_ptr = &a_data[i * a_stride];
    magma_int_t* pivots_working_ptr = &pivots_data[i * pivots_stride];
    magma_int_t* info_working_ptr = &info_data[i];
    magmaLdlHermitian<scalar_t>(
        uplo,
        n,
        a_working_ptr,
        leading_dim,
        pivots_working_ptr,
        info_working_ptr);
  }
  pivots.copy_(pivots_cpu);
  info.copy_(info_cpu);
#endif
}

void ldl_factor_magma(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  if (LD.is_complex()) {
    TORCH_CHECK(
        hermitian,
        "torch.linalg.ldl_factor: complex tensors with hermitian=False flag are not supported with MAGMA backend. ",
        "Currently preferred backend is ",
        at::globalContext().linalgPreferredBackend(),
        ", please set 'default' or 'cusolver' backend with torch.backends.cuda.preferred_linalg_library");
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_magma", [&] {
        apply_ldl_factor_magma<scalar_t>(LD, pivots, info, upper);
      });
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
       { ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
        return;
}
    case at::LinalgBackend::Magma:
       { ldl_factor_magma(LD, pivots, info, upper, hermitian);
        return;
}
    default:
    // By default use cusolver if available and magma otherwise.
    // If cusolver and magma 2.5.4+ are both available and hermitian=true,
    // call magma for complex inputs
#ifdef USE_LINALG_SOLVER
#if AT_MAGMA_ENABLED() && (AT_MAGMA_VERSION >= 254)
      if (LD.is_complex() && hermitian) {
        return ldl_factor_magma(
            LD, pivots, info, upper, hermitian);
      }
#endif
    { ldl_factor_cusolver(
      LD, pivots, info, upper, hermitian);
      return;
    }
#else
      return ldl_factor_magma(LD, pivots, info, upper, hermitian);
#endif
  }
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
  // TODO: It should be possible to add the MAGMA backend for this function when using MAGMA 2.6.0
  // https://bitbucket.org/icl/magma/src/c703d112dcf19eb8c73676cef10888aa2ef73457/ReleaseNotes#lines-48
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_solve: complex tensors with hermitian=True flag are not supported on CUDA.");
  }

  ldl_solve_cusolver(LD, pivots, B, upper);
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_CUDA_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#if !AT_MAGMA_ENABLED()
TORCH_CHECK(false, "cholesky_solve: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t lda = std::max<magma_int_t>(1, n);
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  int info_tmp = 0;
  if (b.dim() == 2) {
    magmaCholeskySolve<scalar_t>(uplo, n, nrhs, A_data, lda,
                                 b_data, lda, &info_tmp);
    info = info_tmp;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");

    scalar_t** A_array;
    scalar_t** b_array;

    ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
    ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      A_array[i] = &A_data[i * A_mat_stride];
      b_array[i] = &b_data[i * b_mat_stride];
    }

    MAGMAQueue magma_queue(b.get_device());

    constexpr int64_t batch_limit = 65535;
    // Compute as many batches of 65535 possible
    // The number of "mini"-batches are floor(batch_size / batch_limit)
    // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
    int64_t mini_batches = batch_size / batch_limit, mini_idx;
    for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
      scalar_t** A_array_cur = &A_array[mini_idx];
      scalar_t** b_array_cur = &b_array[mini_idx];

      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, A_array_cur, lda, b_array_cur, lda,
          info_tmp, batch_limit, magma_queue);

      if (info_tmp != 0) {
        break;
      }
    }

    // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
    // which concisely is equal to batch_size % batch_limit
    if (batch_size % batch_limit != 0 && info_tmp == 0) {
      magmaCholeskySolveBatched<scalar_t>(
          uplo, n, nrhs, &A_array[mini_idx], lda, &b_array[mini_idx], lda,
          info_tmp, batch_size % batch_limit, magma_queue);
    }

    info = info_tmp;
  }
#endif
}

Tensor _cholesky_solve_helper_cuda_magma(const Tensor& self, const Tensor& A, bool upper) {
  int64_t info = 0;
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_solve_cuda", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, info);
  });
  TORCH_CHECK(info == 0, "MAGMA cholesky_solve : invalid argument: ", -info);
  return self_working_copy;
}

// Todo: cusolverDn<T>potrsBatched only supports nrhs == 1 and does not have good performance.
//     Batched cholesky_solve is dispatched to magma.
Tensor _cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
#if defined(USE_LINALG_SOLVER)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
    case at::LinalgBackend::Magma:
      return _cholesky_solve_helper_cuda_magma(self, A, upper);
    default:
      if (batchCount(self) == 1 || !use_magma_) {
        return _cholesky_solve_helper_cuda_cusolver(self, A, upper);
      } else {
        return _cholesky_solve_helper_cuda_magma(self, A, upper);
      }
  }
#else
  return _cholesky_solve_helper_cuda_magma(self, A, upper);
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

static void cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
  _warn_once_magma_deprecation("linalg.eig");
  cholesky_helper_cusolver(input, upper, info);
}

REGISTER_CUDA_DISPATCH(cholesky_stub, &cholesky_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the inverse of a symmetric (Hermitian) positive-definite matrix n-by-n matrix 'input' using the Cholesky solver
This is an in-place routine, content of 'input' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
MAGMA requires 'infos' to reside in CPU memory.
For more information see MAGMA's documentation for POTRS routine.
*/
template <typename scalar_t>
static void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(false, "cholesky_inverse: MAGMA library not found in compilation. Please rebuild with MAGMA.");
#else
  // magmaCholeskyInverse (magma_dpotri_gpu) is slow because internally
  // it transfers data several times between GPU and CPU and calls lapack routine on CPU
  // using magmaCholeskySolveBatched is a lot faster
  // note that magmaCholeskySolve is also slow

  // 'input' is modified in-place we need to clone it and replace with a diagonal matrix
  // for apply_cholesky_solve
  auto input_working_copy = cloneBatchedColumnMajor(input);

  // 'input' tensor has to be a batch of diagonal matrix
  input.fill_(0);
  input.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);

  Tensor result_u, input_u;
  if (input.dim() == 2) {
    // unsqueezing here so that the batched version is used
    result_u = input.unsqueeze(0);
    input_u = input_working_copy.unsqueeze(0);
  } else {
    result_u = input;
    input_u = input_working_copy;
  }

  // magma's potrs_batched doesn't take matrix-wise array of ints as an 'info' argument
  // it returns a single 'magma_int_t'
  // if info = 0 the operation is successful, if info = -i, the i-th parameter had an illegal value.
  int64_t info_tmp = 0;
  apply_cholesky_solve<scalar_t>(result_u, input_u, upper, info_tmp);
  infos.fill_(info_tmp);
#endif
}

// This is a type dispatching helper function for 'apply_cholesky_inverse'
Tensor& cholesky_inverse_kernel_impl_magma(Tensor &result, Tensor& infos, bool upper) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_inverse_out_cuda", [&]{
    apply_cholesky_inverse<scalar_t>(result, infos, upper);
  });
  return result;
}

Tensor& cholesky_inverse_kernel_impl(Tensor &result, Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_cholesky_inverse'
#if defined(USE_LINALG_SOLVER)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      return cholesky_inverse_kernel_impl_cusolver(result, infos, upper);
    case at::LinalgBackend::Magma:
      return cholesky_inverse_kernel_impl_magma(result, infos, upper);
    default:
      if (batchCount(result) == 1 ||
          !use_magma_) {
        return cholesky_inverse_kernel_impl_cusolver(result, infos, upper);
      } else {
        return cholesky_inverse_kernel_impl_magma(result, infos, upper);
      }
  }
#else
  return cholesky_inverse_kernel_impl_magma(result, infos, upper);
#endif

}

REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
  Computes the LU decomposition of a m×n matrix or batch of matrices in 'input' tensor.
  This is an in-place routine, content of 'input', 'pivots', and 'infos' is overwritten.
  This is a "looped" variant for calling single input MAGMA function on batched input.

  Args:
  * `input` - [in] the input matrix for LU decomposition
              [out] the LU decomposition
  * `pivots` - [out] the pivot indices
  * `infos` - [out] error codes, positive values indicate singular matrices
  * `compute_pivots` - controls whether LU is computed with or without pivoting

  For further details, please see the MAGMA documentation for magma_dgetrf_gpu.
*/
template <typename scalar_t>
static void apply_lu_factor_looped_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_MAGMA_ENABLED()
  // This should never be thrown if the calling functions are correct.
  TORCH_CHECK(false, "linalg.lu_factor: PyTorch was not compiled with MAGMA support.");
#else
  // magmaLu and magmaLuNoPiv require infos and pivots tensor to be on CPU
  // the data is later copied back to the appropriate output tensor
  Tensor infos_cpu = at::empty_like(infos, infos.options().device(kCPU).pinned_memory(true));

  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos_cpu.mutable_data_ptr<magma_int_t>();
  auto input_matrix_stride = matrixStride(input);
  auto pivots_stride = pivots.size(-1);
  auto batch_size = batchCount(input);
  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  auto leading_dimension = std::max<magma_int_t>(1, m);

  if (compute_pivots) {
    Tensor pivots_cpu = at::empty_like(pivots, pivots.options().device(kCPU).pinned_memory(true));
    auto pivots_data = pivots_cpu.mutable_data_ptr<magma_int_t>();
    for (decltype(batch_size) i = 0; i < batch_size; i++) {
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      int* pivots_working_ptr = &pivots_data[i * pivots_stride];
      int* infos_working_ptr = &infos_data[i];
      magmaLu<scalar_t>(m, n, input_working_ptr, leading_dimension, pivots_working_ptr, infos_working_ptr);
    }
    pivots.copy_(pivots_cpu);
  } else {
    for (decltype(batch_size) i = 0; i < batch_size; i++) {
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      int* infos_working_ptr = &infos_data[i];
      magmaLuNoPiv<scalar_t>(m, n, input_working_ptr, leading_dimension, infos_working_ptr);
    }
  }
  infos.copy_(infos_cpu);
#endif
}

/*
  Computes the LU decomposition of a m×n matrix or batch of matrices in 'input' tensor.
  This is an in-place routine, content of 'input', 'pivots', and 'infos' is overwritten.
  This is a specialized batched variant, it is expected to be faster than the "looped" version only for small inputs.

  Args:
  * `input` - [in] the input matrix for LU decomposition
              [out] the LU decomposition
  * `pivots` - [out] the pivot indices
  * `infos` - [out] error codes, positive values indicate singular matrices
  * `compute_pivots` - controls whether LU is computed with or without pivoting

  For further details, please see the MAGMA documentation for magma_dgetrf_batched.
*/
template <typename scalar_t>
static void apply_lu_factor_batched_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_factor on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  // There is a bug in lu_factor_batched_magma in MAGMA < 2.5.2, see
  // https://bitbucket.org/icl/magma/issues/13/getrf_batched-kernel-produces-nans-on
  std::tuple<magma_int_t, magma_int_t, magma_int_t> version;
  magma_version(&std::get<0>(version), &std::get<1>(version), &std::get<2>(version));
  const bool magma_batched_buggy = version < std::make_tuple<magma_int_t, magma_int_t, magma_int_t>(2, 5, 2);
  TORCH_CHECK(!magma_batched_buggy, "linalg.lu_factor has buggs on MAGMA < 2.5.2. Please update your MAGMA version to a newer one.");

  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<magma_int_t>();
  auto input_matrix_stride = matrixStride(input);
  magma_int_t batch_size = magma_int_cast(batchCount(input), "batchCount");

  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  auto leading_dimension = std::max<magma_int_t>(1, m);

  scalar_t** input_array;
  ALLOCATE_ARRAY(input_array, scalar_t*, batch_size);

  // Set up array of pointers to matrices
  for (int64_t i = 0; i < batch_size; i++) {
    input_array[i] = &input_data[i * input_matrix_stride];
  }

  // needed to run lu tests in parallel, see https://github.com/pytorch/pytorch/issues/82894 for examples
  // of failures
  c10::cuda::device_synchronize();
  MAGMAQueue magma_queue(input.get_device());

  if (compute_pivots) {
    auto pivots_data = pivots.data_ptr<magma_int_t>();
    auto pivots_stride = pivots.size(-1);
    // fill pivots with ones to avoid memory access violations inside magma kernels
    // magmaLuBatched might not set the values for it
    // see https://github.com/pytorch/pytorch/pull/53064
    pivots.fill_(1);
    magma_int_t** pivots_array;
    ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
      pivots_array[i] = &pivots_data[i * pivots_stride];
    }
    magmaLuBatched<scalar_t>(m, n, input_array, leading_dimension, pivots_array, infos_data, batch_size, magma_queue);
  } else {
    magmaLuNoPivBatched<scalar_t>(m, n, input_array, leading_dimension, infos_data, batch_size, magma_queue);
  }

  // block CPU until all operations on the queue are finished
  // this explicit sync prevents garbage results from the subsequent magmaLuSolveBatched call from a different queue
  magma_queue_sync(magma_queue.get_queue());
#endif
}

static void lu_factor_looped_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_factor_magma_looped", [&]{
    apply_lu_factor_looped_magma<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

static void lu_factor_batched_magma(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_factor_magma_batched", [&]{
    apply_lu_factor_batched_magma<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

static void lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  auto batch_size = batchCount(input);
  (void) batch_size; // Silence unused warning in some builds
  auto m = input.size(-2);
  auto n = input.size(-1);

  const auto lu_factor_magma = [batch_size](const Tensor& input, const Tensor& pivots, const Tensor& infos, const bool compute_pivots) {
    if (batch_size == 1) {
      lu_factor_looped_magma(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_magma(input, pivots, infos, compute_pivots);
    }
  };

  const auto preferred_backend = at::globalContext().linalgPreferredBackend();
#ifdef USE_LINALG_SOLVER
  const auto lu_factor_cusolver = [batch_size, m, n](const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
    if (m != n || (batch_size == 1 || m >= 512)) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
  };

  if (preferred_backend == at::LinalgBackend::Cusolver) {
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
  } else
#endif // ifdef USE_LINALG_SOLVER
  if (preferred_backend == at::LinalgBackend::Magma) {
    lu_factor_magma(input, pivots, infos, compute_pivots);
  } else {  // preferred backend == default
#ifdef USE_LINALG_SOLVER
#if AT_MAGMA_ENABLED()
    // If magma batched is buggy, we use cusolver
    // otherwise, lu_factor just works for square matrices, for non-square matrices magma batched is the fastest
    // otherwise (i.e. for square matrices), we choose between cusolver and magma using a heuristic
    // ROCm: magma_batched is buggy on rocm also. If we are here, we have access to hipSOLVER so always use
    // it instead of magma
#ifdef USE_ROCM
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#else
    if (m == n && (batch_size == 1 || m <= 16 || (m <= 128 && batch_size <= 16))) {
      lu_factor_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_magma(input, pivots, infos, compute_pivots);
    }
#endif // USE_ROCM
#else // !AT_MAGMA_ENABLED
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#endif // AT_MAGMA_ENABLED
#else // !USE_LINALG_SOLVER
    lu_factor_magma(input, pivots, infos, compute_pivots);
#endif // USE_LINALG_SOLVER
  }

  // We return the trivial permutation of pivots starting with 1 (FORTRAN indexing)
  if (!compute_pivots) {
    auto k = std::min(input.size(-2), input.size(-1));
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    pivots.copy_(pivots_tmp);
  }
}

REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if defined(USE_ROCM)
template <typename scalar_t>
static void apply_triangular_solve_batched_magma(const Tensor& A, const Tensor& b, bool left, bool upper, TransposeType transpose, bool unitriangular) {
#if !AT_MAGMA_ENABLED()
TORCH_CHECK(false, "triangular_solve: MAGMA library not found in "
         "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  magma_trans_t trans = to_magma(transpose);
  magma_diag_t diag = unitriangular ? MagmaUnit : MagmaNonUnit;
  magma_side_t side = left ? MagmaLeft : MagmaRight;

  auto A_data = A.data_ptr<scalar_t>();
  auto b_data = b.data_ptr<scalar_t>();
  // This allows to pass rectangular A and b when left = True
  magma_int_t m = magma_int_cast(left ? A.size(-1) : b.size(-2), "m");
  magma_int_t n = magma_int_cast(b.size(-1), "n");
  // magma returns early if m <= 0 || n <= 0 for magmaTriangularSolveBatched
  // magmaTriangularSolve is calling cuBLAS and it prints
  // ** On entry to DTRSM  parameter number 9 had an illegal value
  // so let's use proper lda parameter here
  magma_int_t lda = std::max<magma_int_t>(1, A.size(-2));
  magma_int_t ldb = std::max<magma_int_t>(1, b.size(-2));
  magma_int_t batch_size = magma_int_cast(batchCount(A), "batch_size");

  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
  }

  MAGMAQueue magma_queue(b.get_device());

  constexpr int64_t batch_limit = 65535;
  // Compute as many batches of 65535 as possible
  // The number of "mini"-batches are floor(batch_size / batch_limit)
  // and these cover floor(batch_size / batch_limit) * batch_limit matrix solves
  int64_t mini_batches = batch_size / batch_limit;
  int64_t mini_idx; // this is outside the loop because it is used for the case batch_size % batch_limit != 0
  for (mini_idx = 0; mini_idx < mini_batches * batch_limit; mini_idx += batch_limit) {
    scalar_t** A_array_cur = &A_array[mini_idx];
    scalar_t** b_array_cur = &b_array[mini_idx];

    magmaTriangularSolveBatched<scalar_t>(
        side, uplo, trans, diag, m, n, A_array_cur,
        lda, b_array_cur, ldb, batch_limit, magma_queue);
  }

  // Compute whatever is left = batch_size - floor(batch_size / batch_limit) * batch_limit
  // which concisely is equal to batch_size % batch_limit
  if (batch_size % batch_limit != 0) {
    magmaTriangularSolveBatched<scalar_t>(
        side, uplo, trans, diag, m, n, &A_array[mini_idx],
        lda, &b_array[mini_idx], ldb, batch_size % batch_limit, magma_queue);
  }
#endif
}

void triangular_solve_batched_magma(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve_batched_magma<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}
#endif // defined(USE_ROCM)

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // For batches smaller than 8 and matrix sizes larger than 64x64 cuBLAS forloop is faster than batched version
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_cublas(A, B, left, upper, transpose, unitriangular);
  } else {
#if !AT_MAGMA_ENABLED() || !defined(USE_ROCM)
    triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
#else
    // cuBLAS batched is faster than MAGMA batched up until 512x512, after that MAGMA is faster
    if (A.size(-1) <= 512) {
      triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
    } else {
      triangular_solve_batched_magma(A, B, left, upper, transpose, unitriangular);
    }
#endif // AT_MAGMA_ENABLED() || !defined(USE_ROCM)
  }
}

REGISTER_CUDA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  // TODO: It is possible to implement efficient batched orgqr for small tau (tau.size(-1) <= 32)
  // using MAGMA, however it fails on Windows because of some illegal memory reads inside MAGMA.
  // See discussions in https://github.com/pytorch/pytorch/pull/51348 for comparison of cuSOLVER-MAGMA
  // and Windows failure.
  // For reference here is the MAGMA-based implementation: https://gist.github.com/IvanYashchuk/2db50002c9d3c1462ff769e6410ad983
#ifdef USE_LINALG_SOLVER
  return orgqr_helper_cusolver(result, tau); // cusolver
#else
  TORCH_CHECK(false, "Calling torch.orgqr on a CUDA tensor requires compiling ",
    "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(orgqr_stub, &orgqr_kernel_impl)

void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#ifdef USE_LINALG_SOLVER
  ormqr_cusolver(input, tau, other, left, transpose);
#else
  TORCH_CHECK(false,
      "Calling torch.ormqr on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(ormqr_stub, &ormqr_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void geqrf_kernel(const Tensor& input, const Tensor& tau) {
  _warn_once_magma_deprecation("linalg.qr");
  auto geqrf_cusolver_backend = [](const Tensor& input, const Tensor& tau) {
      // For the benchmarks see
      // https://github.com/pytorch/pytorch/pull/56253#discussion_r622851107
      // TODO: re-eval
      if (input.size(-2) <= 256 && batchCount(input) >= std::max<int64_t>(2, input.size(-2) / 16)) {
        geqrf_batched_cublas(input, tau);
        return;
      } else {
        geqrf_cusolver(input, tau);
        return;
      }
      geqrf_batched_cublas(input, tau);
      return;
  };
  return geqrf_cusolver_backend(input, tau);
}

REGISTER_CUDA_DISPATCH(geqrf_stub, &geqrf_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  _warn_once_magma_deprecation("linalg.eigh");
  linalg_eigh_cusolver(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
}

REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the eigenvalues and eigenvectors of n-by-n matrix 'input'.
This is an in-place routine, content of 'input', 'values', 'vectors' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see MAGMA's documentation for GEEV routine.
*/
#if defined(USE_ROCM) || !(defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702))
template <typename scalar_t>
void apply_magma_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
#if !AT_MAGMA_ENABLED()
TORCH_CHECK(false, "Calling torch.linalg.eig with MAGMA requires compiling PyTorch with MAGMA. "
                   "Either transfer the tensor to the CPU before calling torch.linalg.eig or use cuSolver.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == at::kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == at::kCPU);
  if (compute_eigenvectors) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(vectors.device() == at::kCPU);
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  magma_vec_t jobvr = compute_eigenvectors ? MagmaVec : MagmaNoVec;
  magma_vec_t jobvl = MagmaNoVec;  // only right eigenvectors are computed
  magma_int_t n = magma_int_cast(input.size(-1), "n");
  auto lda = std::max<magma_int_t>(1, n);
  auto batch_size = batchCount(input);
  auto input_matrix_stride = matrixStride(input);
  auto values_stride = values.size(-1);
  auto input_data = input.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<magma_int_t>();
  auto rvectors_data = compute_eigenvectors ? vectors.data_ptr<scalar_t>() : nullptr;
  scalar_t* lvectors_data = nullptr;  // only right eigenvectors are computed
  int64_t ldvr = compute_eigenvectors ? lda : 1;
  int64_t ldvl = 1;

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (input.is_complex()) {
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    rwork = at::empty({lda * 2}, input.options().dtype(real_dtype));
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // call magmaEig once to get the optimal size of work_data
  scalar_t work_query;
  magmaEig<scalar_t, value_t>(jobvl, jobvr, n, input_data, lda, values_data,
    lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

  magma_int_t lwork = std::max<magma_int_t>(1, static_cast<magma_int_t>(real_impl<scalar_t, value_t>(work_query)));
  Tensor work = at::empty({lwork}, input.dtype());
  auto work_data = work.mutable_data_ptr<scalar_t>();

  for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* values_working_ptr = &values_data[i * values_stride];
    scalar_t* rvectors_working_ptr = compute_eigenvectors ? &rvectors_data[i * input_matrix_stride] : nullptr;
    int* info_working_ptr = &infos_data[i];
    magmaEig<scalar_t, value_t>(jobvl, jobvr, n, input_working_ptr, lda, values_working_ptr,
      lvectors_data, ldvl, rvectors_working_ptr, ldvr, work_data, lwork, rwork_data, info_working_ptr);
  }
#endif
}

// MAGMA wrapper: transfers tensors to CPU, calls apply_magma_eig, then copies results back.
void linalg_eig_magma(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors){
  // MAGMA doesn't have GPU interface for the eigendecomposition, and it forces us to transfer to CPU
  auto eigenvalues_cpu = eigenvalues.cpu();
  auto eigenvectors_cpu = eigenvectors.cpu();
  auto infos_cpu = infos.cpu();

  Tensor input_cpu = at::empty(input.sizes(), input.options().device(kCPU));
  input_cpu.transpose_(-2, -1);
  input_cpu.copy_(input);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cuda", [&]{
    apply_magma_eig<scalar_t>(eigenvalues_cpu, eigenvectors_cpu, input_cpu, infos_cpu, compute_eigenvectors);
  });

  eigenvalues.copy_(eigenvalues_cpu);
  eigenvectors.copy_(eigenvectors_cpu);
  infos.copy_(infos_cpu);
}
#endif

void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_cuda());
  // This function calculates the non-symmetric eigendecomposition in-place
  // tensors should be in batched column major memory format
  // the content of eigenvalues, eigenvectors and infos is overwritten by 'linalg_eig_magma' or
  // 'linalg_eig_cusolver_xgeev' both geev routines modify the provided input matrix in-place, therefore we need a copy
#if !defined(USE_ROCM) && defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702)
  _warn_once_magma_deprecation("linalg.eig");
  linalg_eig_cusolver_xgeev(eigenvalues, eigenvectors, input, infos, compute_eigenvectors);
#else
  // hipSolver does not have `geev`
  _warn_once_magma_deprecation("linalg.eig", /*force_cusolver=*/false);
  linalg_eig_magma(eigenvalues, eigenvectors, infos, input, compute_eigenvectors);
#endif
}

REGISTER_CUDA_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const std::optional<std::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& info) {
  _warn_once_magma_deprecation("linalg.svd");
  // svd_cusolver computes V rather than Vh, so we pass a view of Vh.mT
  // and then conjugate Vh in-place
  svd_cusolver(A, full_matrices, compute_uv, driver, U, S, compute_uv ? Vh.mT() : Vh, info);
  if (compute_uv && Vh.is_complex()) {
    Vh._set_conj(!Vh.is_conj());
  }
}

REGISTER_CUDA_DISPATCH(svd_stub, &svd_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

c10::MaybeOwned<Tensor> maybe_expand_lu(const Tensor& B, const Tensor& LU) {
  // B and LU have the same number of dimensions
  if (batchCount(B) != batchCount(LU)) {
        auto n = B.dim();
    auto expand_shape = DimVector(B.sizes().slice(0, n - 2));
    expand_shape.append({LU.size(-2), LU.size(-1)});
    return c10::MaybeOwned<Tensor>::owned(
        cloneBatchedColumnMajor(LU.expand(expand_shape)));
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(LU);
  }
}

c10::MaybeOwned<Tensor> maybe_expand_pivots(const Tensor& B, const Tensor& pivots) {
  // B and pivots have the same number of dimensions
  if (batchCount(B) != batchCount(pivots.unsqueeze(-1))) {
    auto expand_shape = DimVector(B.sizes().slice(0, B.dim() - 2));
    expand_shape.push_back(pivots.size(-1));
    return c10::MaybeOwned<Tensor>::owned(pivots.expand(expand_shape).contiguous());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(pivots);
  }
}

static void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Trivial case. Remove it once `torch.solve` is removed, as linalg.solve already shortcuts this case
  if (B.numel() == 0) {
    return;
  }

  auto b = batchCount(B);
  auto n = LU.size(-2);
  auto k = B.size(-1);
  // heuristics determined from tests discussed in https://github.com/pytorch/pytorch/pull/72935

  // Computes X = U^{-1}L^{-1}P^T B via triangular solves
  // Helps mitigating the bugs in magma
  auto lu_solve_triangular = [n](const Tensor& LU, const Tensor& pivots, const Tensor& B, const TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    // LAPACK / cublas / etc returns the permutation in an odd format
    // Here we transform it to a vector representing a permutation, i.e. a (batch of) vectors st. P(i) = j
    auto perm = at::arange(n, pivots_->options().dtype(kLong)).expand(pivots_->sizes()).contiguous();
    auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .declare_static_shape(pivots_->sizes(), /*squash_dim=*/pivots_->dim() - 1)
      .add_output(perm)
      .add_const_input(*pivots_)
      .build();
    unpack_pivots_stub(pivots_->device().type(), iter, n, n);

    if (trans == TransposeType::NoTranspose) {
      // Get the inverse permutation
      // This is an insertion sort, and it's equivalent to
      // perm = at::argsort(perm);
      // but more parallelisable and O(n), exploiting that perm is a permutation
      auto id_perm = at::arange(n, perm.options()).expand(perm.sizes());
      auto inv_perm = perm.scatter(-1, perm, id_perm);
      // B1 = P^T @ B  (must be done out-of-place as B is both source and target)
      auto B1 = B.scatter(-2, inv_perm.unsqueeze(-1).expand_as(B), B);
      // B = L^{-1} @ B1
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B1, /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
      // B = U^{-1} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, B, /*upper=*/true);
    } else {
      auto LU_H = LU_->mH();
      // B = U^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/false);
      // B = L^{-H} @ B
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), LU_H, B, /*upper=*/true, /*left=*/true, /*unitriangular=*/true);
      // B = P @ B
      B.scatter_(-2, perm.unsqueeze(-1).expand_as(B), B.clone());
    }
  };

  _warn_once_magma_deprecation("linalg.lu_solve");

#ifdef USE_LINALG_SOLVER
  auto lu_solve_batched_cublas_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    lu_solve_batched_cublas(*LU_, *pivots_, B, trans);
  };

  // Preferred Backend
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  if (preferred_backend == at::LinalgBackend::Cusolver) {
    // TODO: Re-eval this condition
    if (b <= 2 && n >= 64) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    }
    return;
  }

  // TODO: Re-eval this heuristic
  // Heuristic
  //if (n == k) {
  // if (k <= 16) batched_cublas
  // else solve_triag
  //} else {
  //if (n <= 8) {
  // batched_cublas
  //} else if (n <= 32) {
  //  b <= 2 looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 64) {
  //  b <= 2 && (k <= 64 || adjoint) looped_cusolver
  //  k <= 8 batched_cusolver
  //  solve_triag
  //} else if (n <= 128) {
  //  if (b <= 2 && k <= 2) looped_cusolver
  //  else if (k <= 2) batched_cusolver
  //  else solve_triag
  //} else { // n > 128
  //  solve_triag
  //}
  //}

  // Particular case when multiplying A^{-1}B where B is square
  // In this case doing two triangular solves is almost always fastest
  if (n == k) {
    if (n <= 16) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
      return;
    }
    lu_solve_triangular(LU, pivots, B, trans);
    return;
  }

  if (n <= 8) {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  } else if (n <= 64) {
    if (b <= 2 && (k <= 64 || trans != TransposeType::NoTranspose || n <= 32)) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else if (k <= 8) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    } else {
      lu_solve_triangular(LU, pivots, B, trans);
    }
  } else if (n <= 128) {
    if (b <= 2 && k <= 2)  {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else if (k <= 2)  {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    } else {
      lu_solve_triangular(LU, pivots, B, trans);
    }
  } else { // n > 128
    lu_solve_triangular(LU, pivots, B, trans);
  }
#else
  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef USE_LINALG_SOLVER
}

REGISTER_CUDA_DISPATCH(lu_solve_stub, &lu_solve_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void linalg_lstsq_gels(const Tensor& A, const Tensor& B, const Tensor& /*infos*/) {
  // The steps for using the QR decomposition for solving least squares problems
  // are outlined here https://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto mn = std::min(m, n);

  // explicitly broadcast the batch dimensions of A
  // TODO: revisit this later to use batch_iterator_with_broadcasting in triangular_solve
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);

  auto tau_shape = A.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = mn;
  Tensor tau = at::empty(tau_shape, A.options());

  if (m >= n) {
    // Step 1: compute QR factorization using geqrf
    geqrf_kernel(A, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {A.size(-2), A.size(-1)});
    Tensor A_expanded = A.expand({A_expand_batch});
    bool is_fortran_contiguous = A_expanded.mT().is_contiguous();
    Tensor A_broadcasted = is_fortran_contiguous ? A_expanded : cloneBatchedColumnMajor(A_expanded);
    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 2: B <- Q^H B
    ormqr_kernel(A_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/true);

    // Step 3: solve R X = B
    triangular_solve_kernel(
        A_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/TransposeType::NoTranspose,
        /*unitriangular=*/false);
  } else { // underdetermined case
    Tensor Ah = cloneBatchedColumnMajor(A.mH());

    // Step 1: compute QR factorization of conjugate transpose of A using geqrf
    geqrf_kernel(Ah, tau);

    // explicitly broadcast the batch dimensions of A
    // we do it after geqrf so that we don't do redundant computations for the same input
    auto A_expand_batch = expand_batch_portion;
    A_expand_batch.insert(A_expand_batch.end(), {Ah.size(-2), Ah.size(-1)});
    Tensor Ah_expanded = Ah.expand({A_expand_batch});
    bool is_fortran_contiguous = Ah_expanded.mT().is_contiguous();
    Tensor Ah_broadcasted = is_fortran_contiguous ? Ah_expanded : cloneBatchedColumnMajor(Ah_expanded);

    // Step 2: R^H Z = B
    const auto trans = Ah_broadcasted.is_complex() ? TransposeType::ConjTranspose
                                                   : TransposeType::Transpose;
    triangular_solve_kernel(
        Ah_broadcasted,
        B,
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/trans,
        /*unitriangular=*/false);

    // B matrix has the size max(m, n) x nrhs
    // triangular_solve_kernel writes its output into the first m rows of B leaving the rest untouched
    // we need to set the rest of the rows to zero so that the multiplication from step 3 is correct
    B.narrow(-2, m, n - m).zero_();

    auto tau_expand_batch = std::move(expand_batch_portion);
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 3: X <- Q Z
    ormqr_kernel(Ah_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/false);
  }
}

void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& /*rank*/, Tensor& /*singular_values*/, Tensor& infos, double /*rcond*/, std::string /*driver_name*/)  {
  auto m = a.size(-2);
  auto n = a.size(-1);

  _warn_once_magma_deprecation("linalg.lstsq");
  // first handle the underdetermined case (m < n)
  // this case is not supported by cuBLAS
  if (m < n) {
    linalg_lstsq_gels(a, b, infos);
  } else { // m >= n
    // On CUDA platform we use either cuBLAS or cuSOLVER here
    // the batched vs looped dispatch is implemented based on the following performance results
    // https://github.com/pytorch/pytorch/pull/54725#issuecomment-832234456
    if (m <= 256 && batchCount(b) >= std::max<int64_t>(2, m / 16)) {
      gels_batched_cublas(a, b, infos);
    } else {
      linalg_lstsq_gels(a, b, infos);
    }
  }
}

REGISTER_CUDA_DISPATCH(lstsq_stub, &lstsq_kernel)


#if defined(BUILD_LAZY_CUDA_LINALG)
struct DispatchInitializer {
  DispatchInitializer() {
    cuda::detail::LinalgDispatch disp{_cholesky_solve_helper_cuda};
    cuda::detail::registerLinalgDispatch(disp);
  };
} initializer;

}  // namespace lazy_linalg
#endif
}  // namespace at::native

#undef ALLOCATE_ARRAY
