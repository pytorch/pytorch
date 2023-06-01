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
#include <ATen/native/BatchLinearAlgebra.h>
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

template<class scalar_t>
void magmaCholesky(
    magma_uplo_t uplo, magma_int_t n, scalar_t* dA,
    magma_int_t ldda, magma_int_t* info);

template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue);

template<class scalar_t>
void magmaTriangularSolveBatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag, magma_int_t m, magma_int_t n,
    scalar_t** dA_array, magma_int_t ldda, scalar_t** dB_array, magma_int_t lddb, magma_int_t batchsize,
    const MAGMAQueue& magma_queue);

template<class scalar_t>
inline magma_int_t magmaGeqrfOptimalBlocksize(magma_int_t m, magma_int_t n);

template<class scalar_t>
void magmaGeqrf(
    magma_int_t m, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    scalar_t* tau, scalar_t* dT, magma_int_t* info, bool is_v2);

template<class scalar_t, class value_t=scalar_t>
void magmaSyevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, scalar_t* dA, magma_int_t ldda,
    value_t* w, scalar_t* wA, magma_int_t ldwa, scalar_t* work, magma_int_t lwork, value_t* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info);

template<class scalar_t, class value_t=scalar_t>
void magmaEig(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n, scalar_t *A, magma_int_t lda,
    scalar_t *w, scalar_t *VL, magma_int_t ldvl,
    scalar_t *VR, magma_int_t ldvr, scalar_t *work, magma_int_t lwork,
    value_t *rwork,
    magma_int_t *info);

template<class scalar_t, class value_t=scalar_t>
void magmaSvd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, scalar_t* A,
    magma_int_t lda, value_t* s, scalar_t* U, magma_int_t ldu,
    scalar_t* Vh, magma_int_t ldvh, scalar_t* work, magma_int_t lwork,
    value_t* rwork,
    magma_int_t* iwork, magma_int_t* info);

template<class scalar_t>
void magmaLuSolve(
    magma_int_t n, magma_int_t nrhs, scalar_t* dA, magma_int_t ldda, magma_int_t* ipiv,
    scalar_t* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans);

template<class scalar_t>
void magmaLuSolveBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans);

template<class scalar_t>
void magmaGels(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    scalar_t* dA, magma_int_t ldda, scalar_t* dB, magma_int_t lddb,
    scalar_t* hwork, magma_int_t lwork, magma_int_t* info);

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

template<>
void magmaCholesky<double>(
    magma_uplo_t uplo, magma_int_t n, double* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dpotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<float>(
    magma_uplo_t uplo, magma_int_t n, float* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_spotrf_gpu(uplo, n, dA, ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zpotrf_gpu(uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholesky<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA,
    magma_int_t ldda, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cpotrf_gpu(uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<double>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<double>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_zpotrf_batched(uplo, n, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaCholeskyBatched<c10::complex<float>>(
    magma_uplo_t uplo, magma_int_t n, c10::complex<float>** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  magma_cpotrf_batched(uplo, n, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, info_array, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

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

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<double>(magma_int_t m, magma_int_t n) {
  return magma_get_dgeqrf_nb(m, n);
}

template<>
inline magma_int_t magmaGeqrfOptimalBlocksize<float>(magma_int_t m, magma_int_t n) {
  return magma_get_sgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_zgeqrf_nb(m, n);
}

template <>
inline magma_int_t magmaGeqrfOptimalBlocksize<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n) {
  return magma_get_cgeqrf_nb(m, n);
}

template<>
void magmaGeqrf<double>(
    magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
    double* tau, double* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_dgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGeqrf<float>(
    magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
    float* tau, float* dT, magma_int_t* info, bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  } else {
    magma_sgeqrf2_gpu(m, n, dA, ldda, tau, info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<double>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<double>* dA,
    magma_int_t ldda,
    c10::complex<double>* tau,
    c10::complex<double>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_zgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        reinterpret_cast<magmaDoubleComplex*>(dT),
        info);
  } else {
    magma_zgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaDoubleComplex*>(dA),
        ldda,
        reinterpret_cast<magmaDoubleComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template <>
void magmaGeqrf<c10::complex<float>>(
    magma_int_t m,
    magma_int_t n,
    c10::complex<float>* dA,
    magma_int_t ldda,
    c10::complex<float>* tau,
    c10::complex<float>* dT,
    magma_int_t* info,
    bool is_v2) {
  MagmaStreamSyncGuard guard;
  if (!is_v2) {
    magma_cgeqrf_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        reinterpret_cast<magmaFloatComplex*>(dT),
        info);
  } else {
    magma_cgeqrf2_gpu(
        m,
        n,
        reinterpret_cast<magmaFloatComplex*>(dA),
        ldda,
        reinterpret_cast<magmaFloatComplex*>(tau),
        info);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, double* dA, magma_int_t ldda,
    double* w, double* wA, magma_int_t ldwa, double* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_dsyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, float* dA, magma_int_t ldda,
    float* w, float* wA, magma_int_t ldwa, float* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  (void)rwork;  // unused
  (void)lrwork;  // unused
  MagmaStreamSyncGuard guard;
  magma_ssyevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<double>* dA, magma_int_t ldda,
    double* w, c10::complex<double>* wA, magma_int_t ldwa, c10::complex<double>* work, magma_int_t lwork, double* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, w, reinterpret_cast<magmaDoubleComplex*>(wA),
      ldwa, reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSyevd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n, c10::complex<float>* dA, magma_int_t ldda,
    float* w, c10::complex<float>* wA, magma_int_t ldwa, c10::complex<float>* work, magma_int_t lwork, float* rwork,
    magma_int_t lrwork, magma_int_t* iwork, magma_int_t liwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cheevd_gpu(
      jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, w, reinterpret_cast<magmaFloatComplex*>(wA),
      ldwa, reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork, iwork, liwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

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

template<>
void magmaSvd<double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, double* A,
    magma_int_t lda, double* s, double* U, magma_int_t ldu,
    double* Vh, magma_int_t ldvh, double* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_dgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, float* A,
    magma_int_t lda, float* s, float* U, magma_int_t ldu,
    float* Vh, magma_int_t ldvh, float* work, magma_int_t lwork,
    float* rwork, magma_int_t* iwork, magma_int_t* info) {
  (void)rwork; // unused
  MagmaStreamSyncGuard guard;
  magma_sgesdd(jobz, m, n, A, lda, s, U, ldu, Vh, ldvh, work, lwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<float>, float>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<float>* A,
    magma_int_t lda, float* s, c10::complex<float>* U, magma_int_t ldu,
    c10::complex<float>* Vh, magma_int_t ldvh, c10::complex<float>* work, magma_int_t lwork,
    float *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgesdd(jobz, m, n, reinterpret_cast<magmaFloatComplex*>(A), lda, s,
                reinterpret_cast<magmaFloatComplex*>(U), ldu,
                reinterpret_cast<magmaFloatComplex*>(Vh), ldvh,
                reinterpret_cast<magmaFloatComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaSvd<c10::complex<double>, double>(
    magma_vec_t jobz, magma_int_t m, magma_int_t n, c10::complex<double>* A,
    magma_int_t lda, double* s, c10::complex<double>* U, magma_int_t ldu,
    c10::complex<double>* Vh, magma_int_t ldvh, c10::complex<double>* work, magma_int_t lwork,
    double *rwork, magma_int_t* iwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgesdd(jobz, m, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, s,
                reinterpret_cast<magmaDoubleComplex*>(U), ldu,
                reinterpret_cast<magmaDoubleComplex*>(Vh), ldvh,
                reinterpret_cast<magmaDoubleComplex*>(work), lwork,
                rwork, iwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<double>(
    magma_int_t n, magma_int_t nrhs, double* dA, magma_int_t ldda, magma_int_t* ipiv,
    double* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_dgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<float>(
    magma_int_t n, magma_int_t nrhs, float* dA, magma_int_t ldda, magma_int_t* ipiv,
    float* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_sgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<double>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_zgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaDoubleComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolve<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>* dA, magma_int_t ldda, magma_int_t* ipiv,
    c10::complex<float>* dB, magma_int_t lddb, magma_int_t* info, magma_trans_t trans) {
  MagmaStreamSyncGuard guard;
  magma_cgetrs_gpu(trans, n, nrhs, reinterpret_cast<magmaFloatComplex*>(dA), ldda, ipiv, reinterpret_cast<magmaFloatComplex*>(dB), lddb, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    double** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  info = magma_dgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    float** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_sgetrs_batched(trans, n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<double>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<double>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<double>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
  info = magma_zgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaDoubleComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaDoubleComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaLuSolveBatched<c10::complex<float>>(
    magma_int_t n, magma_int_t nrhs, c10::complex<float>** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
    c10::complex<float>** dB_array, magma_int_t lddb, magma_int_t& info,
    magma_int_t batchsize, const MAGMAQueue& magma_queue, magma_trans_t trans) {
 info = magma_cgetrs_batched(trans, n, nrhs, reinterpret_cast<magmaFloatComplex**>(dA_array), ldda, dipiv_array, reinterpret_cast<magmaFloatComplex**>(dB_array), lddb, batchsize, magma_queue.get_queue());
 AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<float>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    float* dA, magma_int_t ldda, float* dB, magma_int_t lddb,
    float* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_sgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<double>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    double* dA, magma_int_t ldda, double* dB, magma_int_t lddb,
    double* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_dgels_gpu(trans, m, n, nrhs,
      dA, ldda, dB, lddb,
      hwork, lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<c10::complex<float>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<float>* dA, magma_int_t ldda, c10::complex<float>* dB, magma_int_t lddb,
    c10::complex<float>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_cgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaFloatComplex*>(dA), ldda,
      reinterpret_cast<magmaFloatComplex*>(dB), lddb,
      reinterpret_cast<magmaFloatComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

template<>
void magmaGels<c10::complex<double>>(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    c10::complex<double>* dA, magma_int_t ldda, c10::complex<double>* dB, magma_int_t lddb,
    c10::complex<double>* hwork, magma_int_t lwork, magma_int_t* info) {
  MagmaStreamSyncGuard guard;
  magma_zgels_gpu(trans, m, n, nrhs,
      reinterpret_cast<magmaDoubleComplex*>(dA), ldda,
      reinterpret_cast<magmaDoubleComplex*>(dB), lddb,
      reinterpret_cast<magmaDoubleComplex*>(hwork), lwork, info);
  AT_CUDA_CHECK(cudaGetLastError());
}

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
      return ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
    case at::LinalgBackend::Magma:
      return ldl_factor_magma(LD, pivots, info, upper, hermitian);
    default:
    // By default use cusolver if available and magma otherwise.
    // If cusolver and magma 2.5.4+ are both available and hermitian=true,
    // call magma for complex inputs
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
#if AT_MAGMA_ENABLED() && (AT_MAGMA_VERSION >= 254)
      if (LD.is_complex() && hermitian) {
        return ldl_factor_magma(
            LD, pivots, info, upper, hermitian);
      }
#endif
      return ldl_factor_cusolver(
          LD, pivots, info, upper, hermitian);
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
AT_ERROR("cholesky_solve: MAGMA library not found in "
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
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
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

template <typename scalar_t>
static void apply_cholesky(const Tensor& self, bool upper, const Tensor& info) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.cholesky on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto self_data = self.data_ptr<scalar_t>();
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");
  auto lda = std::max<magma_int_t>(1, n);

  if (self.dim() == 2) {
    // magmaCholesky requires info to be on CPU
    magma_int_t info_cpu = 0;
    magmaCholesky<scalar_t>(uplo, n, self_data, lda, &info_cpu);
    info.fill_(info_cpu);
  } else {
    TORCH_INTERNAL_ASSERT(info.is_cuda());
    auto info_data = info.data_ptr<magma_int_t>();

    // magmaCholeskyBatched supports only upper=false
    uplo = MagmaLower;

    auto self_mat_stride = matrixStride(self);
    magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");

    scalar_t** self_array;

    ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);

    // Set up the created arrays
    for (int64_t i = 0; i < batch_size; i++) {
      self_array[i] = &self_data[i * self_mat_stride];
    }

    MAGMAQueue magma_queue(self.get_device());

    // Compute as many batches of 262140 possible
    // 262140 is the size of the largest batch of matrices that can be run with
    // violating maximum kernel configuration
    // For complex input the batch limit is 65535 (determined experimentally, see https://github.com/pytorch/pytorch/pull/47047#discussion_r516086923 for more information)
    int64_t batch_limit = self.is_complex() ? 65535 : 262140;

    for (int64_t mini_idx = 0; mini_idx < batch_size; mini_idx += batch_limit) {
      int64_t nbatches = std::min(batch_limit, batch_size - mini_idx);
      scalar_t** self_array_cur = &self_array[mini_idx];
      magma_int_t* info_array_cur = &info_data[mini_idx];

      magmaCholeskyBatched<scalar_t>(
        uplo, n, self_array_cur, lda, info_array_cur, nbatches, magma_queue);
    }
  }
#endif
}

void cholesky_helper_magma(const Tensor& input, bool upper, const Tensor& info) {
  Tensor result = input;
  if (input.dim() > 2) {
    // MAGMA's batched cholesky operator has an off-by-one error causing IMA
    // (see https://github.com/pytorch/pytorch/issues/42666). This code is based
    // on the #cloneBatchedColumnMajor function however it pads the input with
    // one extra element utilizing the fact that the resize_as_ method preserves
    // the storage even if it's larger than the new sizes. This way if MAGMA
    // reads off bounds it will still be valid user memory.
    result = at::empty(input.numel() + 1, input.options());
    result.resize_as_(input).transpose_(-2, -1);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.mT().is_contiguous());

    // batched MAGMA doesn't support upper=true
    // we transpose and conjugate the input as a workaround
    result.copy_(upper ? input.mH() : input);
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    input.scalar_type(), "cholesky_cuda", [&] {
      apply_cholesky<scalar_t>(result, upper, info);
    });

  if (input.dim() > 2) {
    // if upper=true we need to tranpose and conjugate the result tensor
    // because the cholesky decomposition is stored in the lower triangular part
    if (upper) {
      input.copy_(result.mH());
    } else {
      input.copy_(result);
    }
  }
}

static void cholesky_kernel(const Tensor& input, const Tensor& info, bool upper) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Cusolver:
      cholesky_helper_cusolver(input, upper, info);
      break;
    case at::LinalgBackend::Magma:
      cholesky_helper_magma(input, upper, info);
      break;
    default:
      if (batchCount(input) == 1 || !use_magma_ || use_cusolver_potrf_batched_) {
        cholesky_helper_cusolver(input, upper, info);
      } else {
        cholesky_helper_magma(input, upper, info);
      }
  }
#else
  cholesky_helper_magma(input, upper, info);
#endif // USE_LINALG_SOLVER
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
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
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

REGISTER_CUDA_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);

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
  AT_ERROR("linalg.lu_factor: PyTorch was not compiled with MAGMA support.");
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
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  const auto lu_factor_cusolver = [batch_size, m, n](const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
    // In CUDA 10.2, lu_factor_looped_cusolver does not finish the computations when the input
    // matrix is exactly singular. The returned pivots contain garbage. This breaks linalg.det
    // Now, batched_cublas does not handle rectangular matrices, so we still dispatch to
    // looped_cusolver even if m != n.
    constexpr bool looped_correct = CUSOLVER_VERSION >= 11100;
    if (m != n || (looped_correct && (batch_size == 1 || m >= 512))) {
      lu_factor_looped_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_cublas(input, pivots, infos, compute_pivots);
    }
  };

  if (preferred_backend == at::LinalgBackend::Cusolver) {
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
  } else
#endif // ifdef USE_LINALG_SOLVER && !USE_ROCM
  if (preferred_backend == at::LinalgBackend::Magma) {
    lu_factor_magma(input, pivots, infos, compute_pivots);
  } else {  // preferred backend == default
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
#if AT_MAGMA_ENABLED()
    // If magma batched is buggy, we use cusolver
    // otherwise, lu_factor just works for square matrices, for non-square matrices magma batched is the fastest
    // otherwise (i.e. for square matrices), we choose between cusolver and magma using a heuristic
    if (m == n && (batch_size == 1 || m <= 16 || (m <= 128 && batch_size <= 16))) {
      lu_factor_cusolver(input, pivots, infos, compute_pivots);
    } else {
      lu_factor_batched_magma(input, pivots, infos, compute_pivots);
    }
#else // USE_LINALG_SOLVER && !AT_MAGMA_ENABLED
    lu_factor_cusolver(input, pivots, infos, compute_pivots);
#endif
#else // !USE_LINALG_SOLVER
    lu_factor_magma(input, pivots, infos, compute_pivots);
#endif
  }

  // We return the trivial permutation of pivots starting with 1 (FORTRAN indexing)
  if (!compute_pivots) {
    auto k = std::min(input.size(-2), input.size(-1));
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    pivots.copy_(pivots_tmp);
  }
}

REGISTER_CUDA_DISPATCH(lu_factor_stub, &lu_factor);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ triangular_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_triangular_solve_batched_magma(const Tensor& A, const Tensor& b, bool left, bool upper, TransposeType transpose, bool unitriangular) {
#if !AT_MAGMA_ENABLED()
AT_ERROR("triangular_solve: MAGMA library not found in "
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

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  // For batches smaller than 8 and matrix sizes larger than 64x64 cuBLAS forloop is faster than batched version
  if (batchCount(A) <= 8 && A.size(-1) >= 64) {
    triangular_solve_cublas(A, B, left, upper, transpose, unitriangular);
  } else {
#if !AT_MAGMA_ENABLED()
    triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
#else
    // cuBLAS batched is faster than MAGMA batched up until 512x512, after that MAGMA is faster
    if (A.size(-1) <= 512) {
      triangular_solve_batched_cublas(A, B, left, upper, transpose, unitriangular);
    } else {
      triangular_solve_batched_magma(A, B, left, upper, transpose, unitriangular);
    }
#endif // AT_MAGMA_ENABLED()
  }
}

REGISTER_CUDA_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ orgqr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  // TODO: It is possible to implement efficient batched orgqr for small tau (tau.size(-1) <= 32)
  // using MAGMA, however it fails on Windows because of some illegal memory reads inside MAGMA.
  // See discussions in https://github.com/pytorch/pytorch/pull/51348 for comparison of cuSOLVER-MAGMA
  // and Windows failure.
  // For reference here is the MAGMA-based implementation: https://gist.github.com/IvanYashchuk/2db50002c9d3c1462ff769e6410ad983
#if defined(USE_LINALG_SOLVER)
  return orgqr_helper_cusolver(result, tau); // cusolver
#else
  TORCH_CHECK(false, "Calling torch.orgqr on a CUDA tensor requires compiling ",
    "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(orgqr_stub, &orgqr_kernel_impl);

void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#if defined(USE_LINALG_SOLVER)
  ormqr_cusolver(input, tau, other, left, transpose);
#else
  TORCH_CHECK(false,
      "Calling torch.ormqr on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#endif
}

REGISTER_CUDA_DISPATCH(ormqr_stub, &ormqr_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ qr ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_geqrf(const Tensor& input, const Tensor& tau) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
    false,
    "Calling torch.geqrf on a CUDA tensor requires compiling ",
    "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else

  magma_int_t m = magma_int_cast(input.size(-2), "m");
  magma_int_t n = magma_int_cast(input.size(-1), "n");

  auto input_data = input.data_ptr<scalar_t>();
  auto input_matrix_stride = matrixStride(input);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto lda = std::max<int>(1, m);

  // magmaGeqrf uses a hybrid CPU-GPU algorithm to compute the elementary reflectors.
  // The driver routine geqrf2_gpu accepts a tensor on the CPU for elementary reflectors.
  Tensor tau_cpu = at::empty(tau.sizes(), tau.options().device(at::kCPU).pinned_memory(true));
  scalar_t* tau_data = tau_cpu.mutable_data_ptr<scalar_t>();
  scalar_t* work_data = nullptr; // workspace is not needed for geqrf2_gpu

  magma_int_t info = 0;
  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual QR and tau
    // MAGMA's geqrf2_gpu function is used, this version has LAPACK-complaint arguments.
    magmaGeqrf<scalar_t>(m, n, input_working_ptr, lda, tau_working_ptr, work_data, &info, /*is_v2=*/true);
    checkMagmaInternalError(info, "geqrf");
  }
  tau.copy_(tau_cpu, /*non_blocking=*/true);
#endif
}

// This is a type dispatching helper function for 'apply_geqrf'
void geqrf_magma(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_magma", [&]{
    apply_geqrf<scalar_t>(input, tau);
  });
}

void geqrf_kernel(const Tensor& input, const Tensor& tau) {
#ifdef CUDART_VERSION
  auto geqrf_cusolver_backend = [](const Tensor& input, const Tensor& tau) {
    #if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
      // For the benchmarks see
      // https://github.com/pytorch/pytorch/pull/56253#discussion_r622851107
      if (input.size(-2) <= 256 && batchCount(input) >= std::max<int64_t>(2, input.size(-2) / 16)) {
        return geqrf_batched_cublas(input, tau);
      } else {
        return geqrf_cusolver(input, tau);
      }
    #endif
      return geqrf_batched_cublas(input, tau);
  };

  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
  // TODO Investigate whether the following magma bug is still occuring.
  // It may be the case that geqrf followed by orgqr is wrong for the magma backend
  // geqrf_magma currently uses geqrf2_gpu
  //
  // We require to perform ?geqrf_gpu again due to this bug in MAGMA:
  // - ?geqrf_gpu allows fast computation of Q via ?orgqr_gpu, but doesn't give R properly.
  // - ?geqrf2_gpu gives correct R, but doesn't allow computation of Q via ?orgqr_gpu
  // Refer to the below link for more details:
  // http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=1015&p=2800&hilit=geqrf_gpu#p2800
    case at::LinalgBackend::Magma:
      return geqrf_magma(input, tau);
    case at::LinalgBackend::Cusolver:
    default:
      return geqrf_cusolver_backend(input, tau);
  }
#else
  return geqrf_magma(input, tau);
#endif
}

REGISTER_CUDA_DISPATCH(geqrf_stub, &geqrf_kernel);

template <typename scalar_t>
static void apply_magma_eigh(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
    false,
    "Calling torch.linalg.eigh/eigvalsh on a CUDA tensor requires compiling ",
    "PyTorch with MAGMA. Please use PyTorch built with MAGMA support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(values.device() == kCPU);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.device() == kCPU);

  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;
  magma_vec_t jobz = compute_eigenvectors ? MagmaVec : MagmaNoVec;

  magma_int_t n = magma_int_cast(vectors.size(-1), "n");
  auto lda = std::max<magma_int_t>(1, n);
  auto batch_size = batchCount(vectors);

  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<magma_int_t>();

  scalar_t* wA;
  ALLOCATE_ARRAY(wA, scalar_t, lda * lda);

  // Run once, first to get the optimum work sizes.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  magma_int_t lwork = -1;
  scalar_t wkopt;
  magma_int_t liwork = -1;
  magma_int_t iwkopt;
  magma_int_t lrwork = -1;
  value_t rwkopt;
  magmaSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_data, lda, values_data,
    wA, lda, &wkopt, lwork, &rwkopt, lrwork, &iwkopt, liwork, infos_data);

  scalar_t* work;
  magma_int_t* iwork;
  lwork = magma_int_cast(std::max<int64_t>(1, real_impl<scalar_t, value_t>(wkopt)), "work_size");
  liwork = magma_int_cast(std::max<int64_t>(1, iwkopt), "iwork_size");
  ALLOCATE_ARRAY(work, scalar_t, lwork);
  ALLOCATE_ARRAY(iwork, magma_int_t, liwork);

  value_t* rwork = nullptr;
  c10::Storage storage_rwork;
  if (vectors.is_complex()) {
    lrwork = magma_int_cast(std::max<int64_t>(1, rwkopt), "rwork_size");
    storage_rwork = pin_memory<value_t>(lrwork);
    rwork = static_cast<value_t*>(storage_rwork.mutable_data());
  }

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    magma_int_t* info_working_ptr = &infos_data[i];
    magmaSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_working_ptr, lda, values_working_ptr,
      wA, lda, work, lwork, rwork, lrwork, iwork, liwork, info_working_ptr);
    // The current behaviour for Linear Algebra functions to raise an error if something goes wrong
    // or input doesn't satisfy some requirement
    // therefore return early since further computations will be wasted anyway
    if (*info_working_ptr != 0) {
      return;
    }
  }
#endif
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eigh ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This is a type dispatch function for 'apply_magma_eigh'
// For small inputs result is computed on CPU
void linalg_eigh_magma(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // MAGMA just calls LAPACK for eigenvectors.size(-1) <= 128
  // See https://bitbucket.org/icl/magma/src/e6fdca447bd402693e8b0b950a898b6879bbcc41/src/zheevd_gpu.cpp?at=master#lines-258
  // in addition lda is ignored breaking 0x0 inputs
  if (eigenvectors.size(-1) > 128) {
    // MAGMA requires eigenvalues and infos tensors to reside on CPU
    Tensor eigenvalues_cpu = eigenvalues.to(kCPU);
    Tensor infos_cpu = infos.to(kCPU);

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      eigenvectors.scalar_type(), "linalg_eigh_magma", [&] {
        apply_magma_eigh<scalar_t>(
            eigenvalues_cpu, eigenvectors, infos_cpu, upper, compute_eigenvectors);
      });

    // Transfer computed by MAGMA results from CPU to GPU
    eigenvalues.copy_(eigenvalues_cpu);
    infos.copy_(infos_cpu);
  } else { // eigenvectors.size(-1) <= 128
    // transfer to CPU, compute the result and copy back to GPU
    // this is faster than going through MAGMA that does the same
    Tensor eigenvalues_cpu = at::empty_like(eigenvalues, eigenvalues.options().device(kCPU));
    if (compute_eigenvectors) {
      Tensor eigenvectors_cpu = at::empty_like(eigenvectors, eigenvectors.options().device(kCPU));
      at::linalg_eigh_out(eigenvalues_cpu, eigenvectors_cpu, eigenvectors.to(kCPU), upper ? "U" : "L");
      eigenvectors.copy_(eigenvectors_cpu);
    } else {
      at::linalg_eigvalsh_out(eigenvalues_cpu, eigenvectors.to(kCPU), upper ? "U" : "L");
    }
    eigenvalues.copy_(eigenvalues_cpu);
  }
}

void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Magma:
      linalg_eigh_magma(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
      break;
    case at::LinalgBackend::Cusolver:
    default:
      linalg_eigh_cusolver(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
#else
  linalg_eigh_magma(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
#endif
}

REGISTER_CUDA_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ linalg_eig ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Computes the eigenvalues and eigenvectors of n-by-n matrix 'input'.
This is an in-place routine, content of 'input', 'values', 'vectors' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see MAGMA's documentation for GEEV routine.
*/
template <typename scalar_t>
void apply_linalg_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
#if !AT_MAGMA_ENABLED()
TORCH_CHECK(false, "Calling torch.linalg.eig on a CUDA tensor requires compiling PyTorch with MAGMA. "
                   "Either transfer the tensor to the CPU before calling torch.linalg.eig or recompile with MAGMA.");
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

// This is a type dispatching helper function for 'apply_linalg_eig'
void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  // This function calculates the non-symmetric eigendecomposition in-place
  // tensors should be in batched column major memory format
  // the content of eigenvalues, eigenvectors and infos is overwritten by 'apply_linalg_eig'

  // apply_linalg_eig modifies the provided input matrix in-place, therefore we need a copy
  // MAGMA doesn't have GPU interface for the eigendecomposition and it forces us to transfer 'input' to CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_cuda());
  Tensor input_working_copy = at::empty(input.sizes(), input.options().device(kCPU));
  input_working_copy.transpose_(-2, -1);  // make input_working_copy to have Fortran contiguous memory layout
  input_working_copy.copy_(input);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cuda", [&]{
    apply_linalg_eig<scalar_t>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
  });
}

REGISTER_CUDA_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ svd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_svd_magma(const Tensor& A,
                            const bool full_matrices,
                            const bool compute_uv,
                            const Tensor& U,
                            const Tensor& S,
                            const Tensor& Vh,
                            const Tensor& info) {
#if !AT_MAGMA_ENABLED()
AT_ERROR("linalg.svd: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  const auto A_data = A.data_ptr<scalar_t>();
  const auto U_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
  const auto S_data = S.data_ptr<value_t>();
  const auto Vh_data = compute_uv ? Vh.data_ptr<scalar_t>() : nullptr;
  const auto info_data = info.data_ptr<magma_int_t>();
  const auto A_stride = matrixStride(A);
  const auto U_stride = compute_uv ? matrixStride(U) : 0;
  const auto S_stride = S.size(-1);
  const auto Vh_stride = compute_uv ? matrixStride(Vh) : 0;
  const auto batchsize = batchCount(A);
  const auto jobz = compute_uv ? (full_matrices ? MagmaAllVec : MagmaSomeVec) : MagmaNoVec;

  const auto m = magma_int_cast(A.size(-2), "m");
  const auto n = magma_int_cast(A.size(-1), "n");
  const auto lda = magma_int_cast(A.strides().end()[-1], "lda");
  const auto ldu = compute_uv ? magma_int_cast(U.strides().end()[-1], "ldu") : magma_int_t{1};
  const auto ldvh = compute_uv ? magma_int_cast(Vh.strides().end()[-1], "ldvh") : magma_int_t{1};

  c10::Storage storage_rwork;
  value_t* rwork = nullptr;
  if (A.is_complex()) {
    auto lrwork = computeLRWorkDim(compute_uv ? (full_matrices ? 'A' : 'S') : 'N', m, n);
    storage_rwork = pin_memory<value_t>(lrwork);
    rwork = static_cast<value_t*>(storage_rwork.mutable_data());
  }

  magma_int_t* iwork;
  ALLOCATE_ARRAY(iwork, magma_int_t, 8 * std::min(m, n));

  // Query svd for the optimal lwork size
  magma_int_t lwork = -1;
  {
    scalar_t wkopt = 1; // MAGMA might not set the value for the optimal workspace therefore use 1 as the default value
    magmaSvd<scalar_t, value_t>(jobz, m, n,
                                A_data, lda,
                                S_data,
                                compute_uv ? U_data : nullptr, ldu,
                                compute_uv ? Vh_data : nullptr, ldvh,
                                &wkopt, lwork, rwork, iwork, info_data);
    lwork = magma_int_cast(real_impl<scalar_t, value_t>(wkopt), "work_size");
  }
  scalar_t* work;
  ALLOCATE_ARRAY(work, scalar_t, lwork);

  for (int64_t i = 0; i < batchsize; i++) {
    // Compute S, U (optionally), Vh (optionally)
    magmaSvd<scalar_t, value_t>(jobz, m, n,
                                A_data + i * A_stride, lda,
                                S_data + i * S_stride,
                                compute_uv ? U_data + i * U_stride : nullptr, ldu,
                                compute_uv ? Vh_data + i * Vh_stride : nullptr, ldvh,
                                work, lwork, rwork, iwork,
                                info_data + i);
  }
#endif
}

void svd_magma(const Tensor& A,
               const bool full_matrices,
               const bool compute_uv,
               const Tensor& U,
               const Tensor& S,
               const Tensor& Vh,
               const Tensor& info) {
  // A is on GPU and may not have the right strides.
  // We copy it into CPU with the correct strides and in pinned_memory as MAGMA moves things between CPU and GPU
  const auto A_ = A.mT()
                   .to(A.options()
                        .device(kCPU)
                        .memory_format(at::MemoryFormat::Contiguous)
                        .pinned_memory(true))
                   .mT();
  // U, S, Vh, info are the right size and strides, but are on GPU
  // We copy them into CPU in pinned_memory
  const auto empty_like_cpu = [](const Tensor& t) {
    return at::empty_like(t, t.options().device(kCPU).pinned_memory(true));
  };
  auto U_ = compute_uv ? empty_like_cpu(U) : Tensor{};
  auto S_ = empty_like_cpu(S);
  auto Vh_ = compute_uv ? empty_like_cpu(Vh) : Tensor{};
  auto info_ = empty_like_cpu(info);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda", [&] {
    apply_svd_magma<scalar_t>(A_, full_matrices, compute_uv, U_, S_, Vh_, info_);
  });

  // Copy from CPU back to CUDA
  // We can do a non_blocking copy, as there is an unconditional check of the infos in
  // the calling function
  if (compute_uv) {
    U.copy_(U_, /*non_blocking*/true);
    Vh.copy_(Vh_, /*non_blocking*/true);
  }
  S.copy_(S_, /*non_blocking*/true);
  info.copy_(info, /*non_blocking*/true);
}

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const c10::optional<c10::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& info) {
#if defined(USE_LINALG_SOLVER)
  // We always use cuSOLVER unless the user has specified they want to use MAGMA
  bool use_magma = at::globalContext().linalgPreferredBackend() == at::LinalgBackend::Magma;
#ifdef USE_ROCM
  // However for current hipSOLVER, MAGMA is preferred for larger matrices due to
  // performance. Here are a few performance numbers on MI200, ROCM 5.3.22000:
  //    Size       MAGMA     hipSOLVER
  //  100x100      0.127s      0.428s
  //  200x200      0.113s      3.35s
  //  300x300      0.111s      10.9s
  //  400x400      0.126s      25.9s
  //  500x500      0.146s      > 10 minutes, have to kill with SIGTERM
  //
  // TODO: Fix this when hipSOLVER has better performance numbers
  use_magma = use_magma || (A.size(-1) >= 50 && A.size(-2) >= 50);
#endif
  if (use_magma) {
    svd_magma(A, full_matrices, compute_uv, U, S, Vh, info);
  } else {
    // svd_cusolver computes V rather than Vh, so we pass a view of Vh.mT
    // and then conjugate Vh in-place
    svd_cusolver(A, full_matrices, compute_uv, driver, U, S, compute_uv ? Vh.mT() : Vh, info);
    if (compute_uv && Vh.is_complex()) {
      Vh._set_conj(!Vh.is_conj());
    }
  }
#else
  svd_magma(A, full_matrices, compute_uv, U, S, Vh, info);
#endif
}

REGISTER_CUDA_DISPATCH(svd_stub, &svd_kernel)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lu_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
  Solves the matrix equation A X = B
  X and B are n-by-nrhs matrices, A is represented using the LU factorization.
  This is an in-place routine, content of `B` is overwritten.
  This is a "looped" variant for calling single input MAGMA function on batched input.

  Args:
  * `LU` - [in] the LU factorization of matrix A (see at::linalg_lu_factor)
  * `pivots` - [in] the pivot indices (see at::linalg_lu_factor)
  * `B` -  [in] the right hand side matrix B
           [out] the solution matrix X

  For further details, please see the MAGMA documentation for magma_dgetrs_gpu.
*/
template <typename scalar_t>
static void apply_lu_solve_looped_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  auto trans = to_magma(transpose);
  auto b_data = B.data_ptr<scalar_t>();
  auto lu_data = LU.data_ptr<scalar_t>();

  // MAGMA requires pivots to be a CPU tensor
  Tensor pivots_cpu = pivots.cpu();
  auto pivots_data = pivots_cpu.data_ptr<magma_int_t>();

  auto b_stride = matrixStride(B);
  auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
  auto pivots_stride = pivots_cpu.dim() > 1 ? pivots_cpu.stride(-2) : 0;
  auto batch_size = batchCount(B);

  magma_int_t n = magma_int_cast(LU.size(-2), "n");
  magma_int_t nrhs = magma_int_cast(B.size(-1), "nrhs");
  auto leading_dimension = std::max<magma_int_t>(1, n);

  // LU and pivots tensors can be broadcasted to B
  // here we construct a helper indexing tensor to linearly index into lu and pivots
  IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
  IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
  BroadcastLinearIndices lu_index(
      batchCount(LU), lu_batch_shape, b_batch_shape);

  int info = 0;
  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    int64_t lu_index_i = lu_index(i);
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    scalar_t* lu_working_ptr = &lu_data[lu_index_i * lu_stride];
    int* pivots_working_ptr = &pivots_data[lu_index_i * pivots_stride];

    magmaLuSolve<scalar_t>(n, nrhs, lu_working_ptr, leading_dimension, pivots_working_ptr, b_working_ptr, leading_dimension, &info, trans);

    // info from magmaLuSolve only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

/*
  Solves the matrix equation A X = B
  X and B are n-by-nrhs matrices, A is represented using the LU factorization.
  This is an in-place routine, content of `B` is overwritten.
  This is a specialized batched variant, it is expected to be faster than the "looped" version only for small inputs.

  Args:
  * `lu` - [in] the LU factorization of matrix A (see at::linalg_lu_factor)
  * `pivots` - [in] the pivot indices (see at::linalg_lu_factor)
  * `B` -  [in] the right hand side matrix B
           [out] the solution matrix X

  For further details, please see the MAGMA documentation for magma_dgetrs_batched.
*/
template <typename scalar_t>
static void apply_lu_solve_batched_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CUDA tensor requires compiling ",
      "PyTorch with MAGMA. Please rebuild with MAGMA.");
#else
  TORCH_INTERNAL_ASSERT(batchCount(B) == batchCount(LU), "batch_size of LU and B must be the same");
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(pivots.unsqueeze(-1)), "batch_size of LU and pivots must be the same");
  auto trans = to_magma(transpose);
  auto b_data = B.data_ptr<scalar_t>();
  auto lu_data = LU.data_ptr<scalar_t>();

  magma_int_t n = magma_int_cast(LU.size(-2), "n");
  magma_int_t nrhs = magma_int_cast(B.size(-1), "nrhs");
  auto leading_dimension = std::max<magma_int_t>(1, n);

  auto pivots_data = pivots.data_ptr<magma_int_t>();

  auto b_stride = matrixStride(B);
  auto lu_stride = matrixStride(LU);
  auto pivots_stride = pivots.size(-1);
  magma_int_t batch_size = magma_int_cast(batchCount(B), "batchCount");

  magma_int_t** pivots_array;
  scalar_t** lu_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(pivots_array, magma_int_t*, batch_size);
  ALLOCATE_ARRAY(lu_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size);

  for (int64_t i = 0; i < batch_size; i++) {
    pivots_array[i] = &pivots_data[i * pivots_stride];
    b_array[i] = &b_data[i * b_stride];
    lu_array[i] = &lu_data[i * lu_stride];
  }

  MAGMAQueue magma_queue(B.get_device());

  // Compute the result in batches of 65535
  // that is the maximum allowed number for batch_size in MAGMA
  constexpr int64_t batch_limit = 65535;

  for (int64_t mini_idx = 0; mini_idx < batch_size; mini_idx += batch_limit) {
    int64_t nbatches = std::min(batch_limit, batch_size - mini_idx);
    scalar_t** lu_array_cur = &lu_array[mini_idx];
    scalar_t** b_array_cur = &b_array[mini_idx];
    magma_int_t** pivots_array_cur = &pivots_array[mini_idx];

    int info;
    magmaLuSolveBatched<scalar_t>(
        n, nrhs, lu_array_cur, leading_dimension,
        pivots_array_cur, b_array_cur, leading_dimension,
        info, nbatches, magma_queue, trans);

    // info from magmaLuSolveBatched only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

static void lu_solve_batched_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // There is a bug in MAGMA when TransposeType is transpose or conj-transpose.
  TORCH_INTERNAL_ASSERT(trans == TransposeType::NoTranspose);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_batched_magma", [&]{
    apply_lu_solve_batched_magma<scalar_t>(LU, pivots, B, trans);
  });
}

static void lu_solve_looped_magma(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_looped_magma", [&]{
    apply_lu_solve_looped_magma<scalar_t>(LU, pivots, B, trans);
  });
}

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
  // magma implementation of LU solve cannot handle a b tensor with last dim > 1024
  // See https://bitbucket.org/icl/magma/issues/19/dgesv_batched-dgetrs_batched-fails-for
  bool over_batched_magma_dim_limit = k > 1024;
  // heuristics determined from tests dicussed in https://github.com/pytorch/pytorch/pull/72935

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
      .add_input(*pivots_)
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
      at::linalg_solve_triangular_out(const_cast<Tensor&>(B), *LU_, std::move(B1), /*upper=*/false, /*left=*/true, /*unitriangular=*/true);
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

#ifdef CUDART_VERSION
  auto lu_solve_batched_cublas_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    lu_solve_batched_cublas(*LU_, *pivots_, B, trans);
  };
#endif

  auto lu_solve_batched_magma_fn = [](const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
    auto LU_ = maybe_expand_lu(B, LU);
    auto pivots_ = maybe_expand_pivots(B, pivots);
    lu_solve_batched_magma(*LU_, *pivots_, B, trans);
  };


  // Preferred Backend
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  if (preferred_backend == at::LinalgBackend::Cusolver) {
    if (b <= 2 && n >= 64) {
      lu_solve_looped_cusolver(LU, pivots, B, trans);
    } else {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
    }
    return;
  } else
#endif // ifdef USE_LINALG_SOLVER && !USE_ROCM
  if (preferred_backend == at::LinalgBackend::Magma) {
    // Looped magma is very slow, but batched magma is buggy in these two cases
    if (!over_batched_magma_dim_limit && trans == TransposeType::NoTranspose) {
      lu_solve_batched_magma_fn(LU, pivots, B, trans);
    }
    else {
      lu_solve_looped_magma(LU, pivots, B, trans);
    }
    return;
  }

  // Heuristic
  //if (n == k) {
  // if (k <= 16) batched_cublas
  // else solve_triag
  //} else {
  //if (n <= 8) {
  // if (k >= 256 && NoTranspose) batched_magma
  // else batched_cusolver
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
#ifdef CUDART_VERSION
    if (n <= 16) {
      lu_solve_batched_cublas_fn(LU, pivots, B, trans);
      return;
    }
#endif
    lu_solve_triangular(LU, pivots, B, trans);
    return;
  }

#ifdef CUDART_VERSION
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
if (n <= 8) {
  if (use_magma_ && !over_batched_magma_dim_limit && trans == TransposeType::NoTranspose && k >= 256) {
    lu_solve_batched_magma_fn(LU, pivots, B, trans);
  } else {
    lu_solve_batched_cublas_fn(LU, pivots, B, trans);
  }
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
#endif // ifdef USE_LINALG_SOLVER && !USE_ROCM
#else  // No cublas or cusolver
  // lu_solve_triangular is almost always best
  lu_solve_triangular(LU, pivots, B, trans);
#endif // ifdef CUDART_VERSION
}

REGISTER_CUDA_DISPATCH(lu_solve_stub, &lu_solve_kernel);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ lstsq ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_gels(const Tensor& a, Tensor& b, Tensor& infos) {
#if !AT_MAGMA_ENABLED()
  TORCH_CHECK(false, "torch.linalg.lstsq: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto trans = MagmaNoTrans;
  auto m = magma_int_cast(a.size(-2), "m");
  auto n = magma_int_cast(a.size(-1), "n");

  TORCH_CHECK(
    m >= n,
    "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA");

  auto nrhs = magma_int_cast(b.size(-1), "nrhs");
  auto ldda = std::max<magma_int_t>(1, m);
  auto lddb = std::max<magma_int_t>(1, std::max(m, n));
  auto nb = magmaGeqrfOptimalBlocksize<scalar_t>(m, n);
  auto lwork = (m - n + nb) * (nrhs + nb) + nrhs * nb;
  Tensor hwork = at::empty({static_cast<int64_t>(lwork)}, a.scalar_type());
  auto* hwork_ptr = hwork.mutable_data_ptr<scalar_t>();

  // MAGMA requires infos tensor to live on CPU
  infos = infos.to(at::kCPU);
  auto infos_data = infos.data_ptr<magma_int_t>();

  batch_iterator_with_broadcasting<scalar_t>(a, b,
    [&](scalar_t* a_working_ptr, scalar_t* b_working_ptr,
      int64_t a_linear_batch_idx) {
      magma_int_t* infos_working_ptr = &infos_data[a_linear_batch_idx];
      magmaGels<scalar_t>(trans, m, n, nrhs,
        a_working_ptr, ldda, b_working_ptr, lddb,
        hwork_ptr, lwork, infos_working_ptr);
    }
  );
#endif
}

void gels_magma(const Tensor& a, Tensor& b, Tensor& infos) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "gels_magma", [&] {
    apply_gels<scalar_t>(a, b, infos);
  });
}

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
        const_cast<Tensor&>(A_broadcasted),
        const_cast<Tensor&>(B),
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
        const_cast<Tensor&>(Ah_broadcasted),
        const_cast<Tensor&>(B),
        /*left=*/true,
        /*upper=*/true,
        /*transpose=*/trans,
        /*unitriangular=*/false);

    // B matrix has the size max(m, n) x nrhs
    // triangular_solve_kernel writes its output into the first m rows of B leaving the rest untouched
    // we need to set the rest of the rows to zero so that the multiplication from step 3 is correct
    B.narrow(-2, m, n - m).zero_();

    auto tau_expand_batch = expand_batch_portion;
    tau_expand_batch.push_back(tau.size(-1));
    Tensor tau_broadcasted = tau.expand({tau_expand_batch}).contiguous();

    // Step 3: X <- Q Z
    ormqr_kernel(Ah_broadcasted, tau_broadcasted, B, /*left=*/true, /*transpose=*/false);
  }
}

void gels_looped(const Tensor& a, Tensor& b, Tensor& infos) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
  auto preferred_backend = at::globalContext().linalgPreferredBackend();
  switch (preferred_backend) {
    case at::LinalgBackend::Magma:
      return gels_magma(a, b, infos);
    case at::LinalgBackend::Cusolver:
    default:
      // linalg_lstsq_gels is a generic function that is implemented using
      // geqrf_stub, ormqr_stub, and triangular_solve_stub
      // It dispatches to cuSOLVER for CUDA inputs if USE_LINALG_SOLVER is defined
      return linalg_lstsq_gels(a, b, infos);
  }
#else
  return gels_magma(a, b, infos);
#endif
}

void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& /*rank*/, Tensor& /*singular_values*/, Tensor& infos, double /*rcond*/, std::string /*driver_name*/)  {
  auto m = a.size(-2);
  auto n = a.size(-1);

  // first handle the underdetermined case (m < n)
  // this case is not supported by MAGMA or cuBLAS
  if (m < n) {
#if defined(USE_LINALG_SOLVER) && !defined(USE_ROCM)
    linalg_lstsq_gels(a, b, infos);
#else
    TORCH_CHECK(
        false,
        "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA. ",
        "Please rebuild with cuSOLVER.");
#endif
  } else { // m >= n
#if !AT_ROCM_ENABLED()
    // On CUDA platform we use either cuBLAS or cuSOLVER here
    // the batched vs looped dispatch is implemented based on the following performance results
    // https://github.com/pytorch/pytorch/pull/54725#issuecomment-832234456
    if (m <= 256 && batchCount(b) >= std::max<int64_t>(2, m / 16)) {
      gels_batched_cublas(a, b, infos);
    } else {
      gels_looped(a, b, infos);
    }
#else
    // On ROCm platform we can only use MAGMA here
    // If MAGMA is not available, an error will be thrown
    gels_magma(a, b, infos);
#endif // !AT_ROCM_ENABLED()
  }
}

REGISTER_CUDA_DISPATCH(lstsq_stub, &lstsq_kernel);


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
