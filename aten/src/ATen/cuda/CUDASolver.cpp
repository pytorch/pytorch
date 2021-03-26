#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDASolver.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifdef CUDART_VERSION

namespace at {
namespace cuda {
namespace solver {

const char* cusolverGetErrorMessage(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCES";
    case CUSOLVER_STATUS_NOT_INITIALIZED:             return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:                return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:               return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:               return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:            return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:              return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:                                          return "Unknown cusolver error number";
  }
}

template <>
void getrf<double>(
    cusolverDnHandle_t handle, int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<float>(
    cusolverDnHandle_t handle, int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(dataPtr.get()), ipiv, info));
}

template <>
void getrf<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<double>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex) * lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(dA),
      ldda,
      static_cast<cuDoubleComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrf<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m,
    int n,
    c10::complex<float>* dA,
    int ldda,
    int* ipiv,
    int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(
      handle, m, n, reinterpret_cast<cuComplex*>(dA), ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex) * lwork);
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrf(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex*>(dA),
      ldda,
      static_cast<cuComplex*>(dataPtr.get()),
      ipiv,
      info));
}

template <>
void getrs<double>(
    cusolverDnHandle_t handle, int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(
    handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<float>(
    cusolverDnHandle_t handle, int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(
    handle, CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<double>* dA,
    int lda,
    int* ipiv,
    c10::complex<double>* ret,
    int ldb,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrs(
      handle,
      CUBLAS_OP_N,
      n,
      nrhs,
      reinterpret_cast<cuDoubleComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<cuDoubleComplex*>(ret),
      ldb,
      info));
}

template <>
void getrs<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int n,
    int nrhs,
    c10::complex<float>* dA,
    int lda,
    int* ipiv,
    c10::complex<float>* ret,
    int ldb,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrs(
      handle,
      CUBLAS_OP_N,
      n,
      nrhs,
      reinterpret_cast<cuComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(ret),
      ldb,
      info));
}


template<>
void gesvdj<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    static_cast<cuComplex*>(dataPtr.get()),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, int *info, gesvdjInfo_t params
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, &lwork, params));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    lwork, info, params));
}


template<>
void gesvdjBatched<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(float)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<float*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(double)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdjBatched(
    handle, jobz, m, n, A, lda, S, U, ldu, V, ldv,
    static_cast<double*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    static_cast<cuComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}

template<>
void gesvdjBatched<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, int *info, gesvdjInfo_t params, int batchSize
) {
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched_bufferSize(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, &lwork, params, batchSize));

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(cuDoubleComplex)*lwork);

  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdjBatched(
    handle, jobz, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    static_cast<cuDoubleComplex*>(dataPtr.get()),
    lwork, info, params, batchSize));
}


template<>
void potrf<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* info
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrf(
    handle, uplo, n, A, lda, work, lwork, info));
}

template<>
void potrf<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* info
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrf(
    handle, uplo, n, A, lda, work, lwork, info));
}

template<>
void potrf<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<float>* A, int lda, c10::complex<float>* work, int lwork, int* info
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrf(
    handle,
    uplo,
    n,
    reinterpret_cast<cuComplex*>(A),
    lda,
    reinterpret_cast<cuComplex*>(work),
    lwork,
    info));
}

template<>
void potrf<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<double>* A, int lda, c10::complex<double>* work, int lwork, int* info
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrf(
    handle,
    uplo,
    n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda,
    reinterpret_cast<cuDoubleComplex*>(work),
    lwork,
    info));
}


template<>
void potrf_buffersize<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* A, int lda, int* lwork
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template<>
void potrf_buffersize<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* A, int lda, int* lwork
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork));
}

template<>
void potrf_buffersize<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<float>* A, int lda, int* lwork
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrf_bufferSize(
    handle, uplo, n,
    reinterpret_cast<cuComplex*>(A),
    lda, lwork));
}

template<>
void potrf_buffersize<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<double>* A, int lda, int* lwork
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrf_bufferSize(
    handle, uplo, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, lwork));
}


template<>
void potrfBatched<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float** A, int lda, int* info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template<>
void potrfBatched<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double** A, int lda, int* info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize));
}

template<>
void potrfBatched<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<float>** A, int lda, int* info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrfBatched(
    handle, uplo, n,
    reinterpret_cast<cuComplex**>(A),
    lda, info, batchSize));
}

template<>
void potrfBatched<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, c10::complex<double>** A, int lda, int* info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrfBatched(
    handle, uplo, n,
    reinterpret_cast<cuDoubleComplex**>(A),
    lda, info, batchSize));
}


template <>
void orgqr_buffersize<float>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const float* A, int lda,
    const float* tau, int* lwork) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<double>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const double* A, int lda,
    const double* tau, int* lwork) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork));
}

template <>
void orgqr_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<float>* A, int lda,
    const c10::complex<float>* tau, int* lwork) {
  TORCH_CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau), lwork));
}

template <>
void orgqr_buffersize<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    const c10::complex<double>* A, int lda,
    const c10::complex<double>* tau, int* lwork) {
  TORCH_CUSOLVER_CHECK(cusolverDnZungqr_bufferSize(
      handle,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau), lwork));
}

template <>
void orgqr<float>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    float* A, int lda,
    const float* tau,
    float* work, int lwork,
    int* devInfo) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<double>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    double* A, int lda,
    const double* tau,
    double* work, int lwork,
    int* devInfo) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}

template <>
void orgqr<c10::complex<float>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<float>* A, int lda,
    const c10::complex<float>* tau,
    c10::complex<float>* work, int lwork,
    int* devInfo) {
  TORCH_CUSOLVER_CHECK(cusolverDnCungqr(
      handle,
      m, n, k,
      reinterpret_cast<cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(work), lwork,
      devInfo));
}

template <>
void orgqr<c10::complex<double>>(
    cusolverDnHandle_t handle,
    int m, int n, int k,
    c10::complex<double>* A, int lda,
    const c10::complex<double>* tau,
    c10::complex<double>* work, int lwork,
    int* devInfo) {
  TORCH_CUSOLVER_CHECK(cusolverDnZungqr(
      handle,
      m, n, k,
      reinterpret_cast<cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(work), lwork,
      devInfo));
}

#ifdef USE_CUSOLVER_64_BIT

template<> cudaDataType get_cusolver_datatype<float>() { return CUDA_R_32F; }
template<> cudaDataType get_cusolver_datatype<double>() { return CUDA_R_64F; }
template<> cudaDataType get_cusolver_datatype<c10::complex<float>>() { return CUDA_C_32F; }
template<> cudaDataType get_cusolver_datatype<c10::complex<double>>() { return CUDA_C_64F; }

void xpotrf_buffersize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost
  ));
}

void xpotrf(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost,
    int *info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrf(
    handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info
  ));
}


#endif // USE_CUSOLVER_64_BIT

} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
