#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/cuda/linalg/CUDASolver.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/macros/Export.h>


namespace at::cuda::solver {

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
    cusolverDnHandle_t handle, int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info, cublasOperation_t trans) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
}

template <>
void getrs<float>(
    cusolverDnHandle_t handle, int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info, cublasOperation_t trans) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(
    handle, trans, n, nrhs, dA, lda, ipiv, ret, ldb, info));
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
    int* info,
    cublasOperation_t trans) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgetrs(
      handle,
      trans,
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
    int* info,
    cublasOperation_t trans) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgetrs(
      handle,
      trans,
      n,
      nrhs,
      reinterpret_cast<cuComplex*>(dA),
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(ret),
      ldb,
      info));
}

template <>
void sytrf_bufferSize<double>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<float>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZsytrf_bufferSize(
      handle, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
}

template <>
void sytrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCsytrf_bufferSize(
      handle, n, reinterpret_cast<cuComplex*>(A), lda, lwork));
}

template <>
void sytrf<double>(CUDASOLVER_SYTRF_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<float>(CUDASOLVER_SYTRF_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo));
}

template <>
void sytrf<c10::complex<double>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      ipiv,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      devInfo));
}

template <>
void sytrf<c10::complex<float>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCsytrf(
      handle,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      ipiv,
      reinterpret_cast<cuComplex*>(work),
      lwork,
      devInfo));
}

template<>
void gesvd_buffersize<float>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<double>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<c10::complex<float>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvd_bufferSize(handle, m, n, lwork));
}

template<>
void gesvd_buffersize<c10::complex<double>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(handle, m, n, lwork));
}


template<>
void gesvd<float>(CUDASOLVER_GESVD_ARGTYPES(float, float)) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}

template<>
void gesvd<double>(CUDASOLVER_GESVD_ARGTYPES(double, double)) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvd(
      handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info));
}


template<>
void gesvd<c10::complex<float>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<float>, float)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<cuComplex*>(A),
      lda, S,
      reinterpret_cast<cuComplex*>(U),
      ldu,
      reinterpret_cast<cuComplex*>(VT),
      ldvt,
      reinterpret_cast<cuComplex*>(work),
      lwork, rwork, info
  ));
}

template<>
void gesvd<c10::complex<double>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<double>, double)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvd(
      handle, jobu, jobvt, m, n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda, S,
      reinterpret_cast<cuDoubleComplex*>(U),
      ldu,
      reinterpret_cast<cuDoubleComplex*>(VT),
      ldvt,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork, rwork, info
  ));
}


template<>
void gesvdj_buffersize<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float *A, int lda, float *S,
    float *U, int ldu, float *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

template<>
void gesvdj_buffersize<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double *A, int lda, double *S,
    double *U, int ldu, double *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params));
}

template<>
void gesvdj_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float> *A, int lda, float *S,
    c10::complex<float> *U, int ldu, c10::complex<float> *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda,
    S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv, lwork, params));
}

template<>
void gesvdj_buffersize<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<double> *A, int lda, double *S,
    c10::complex<double> *U, int ldu, c10::complex<double> *V, int ldv, int *lwork, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda,
    S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, lwork, params));
}


template<>
void gesvdj<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U,
    int ldu, float *V, int ldv, float* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

template<>
void gesvdj<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U,
    int ldu, double *V, int ldv, double* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params));
}

template<>
void gesvdj<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<float>* A, int lda, float* S, c10::complex<float>* U,
    int ldu, c10::complex<float> *V, int ldv, c10::complex<float>* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, S,
    reinterpret_cast<cuComplex*>(U),
    ldu,
    reinterpret_cast<cuComplex*>(V),
    ldv,
    reinterpret_cast<cuComplex*>(work),
    lwork, info, params));
}

template<>
void gesvdj<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, c10::complex<double>* A, int lda, double* S, c10::complex<double>* U,
    int ldu, c10::complex<double> *V, int ldv, c10::complex<double>* work, int lwork, int *info, gesvdjInfo_t params
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdj(
    handle, jobz, econ, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, S,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv,
    reinterpret_cast<cuDoubleComplex*>(work),
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


// ROCM does not implement gesdva yet
#ifdef CUDART_VERSION
template<>
void gesvdaStridedBatched_buffersize<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, double *A, int lda, long long int strideA,
    double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<float> *A, int lda, long long int strideA,
    float *S, long long int strideS, c10::complex<float> *U, int ldu, long long int strideU,
    c10::complex<float> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<cuComplex*>(U),
    ldu, strideU,
    reinterpret_cast<cuComplex*>(V),
    ldv, strideV, lwork, batchSize
  ));
}

template<>
void gesvdaStridedBatched_buffersize<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<double> *A, int lda, long long int strideA,
    double *S, long long int strideS, c10::complex<double> *U, int ldu, long long int strideU,
    c10::complex<double> *V, int ldv, long long int strideV,
    int *lwork, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdaStridedBatched_bufferSize(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu, strideU,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, strideV, lwork, batchSize
  ));
}


template<>
void gesvdaStridedBatched<float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, float *A, int lda, long long int strideA,
    float *S, long long int strideS, float *U, int ldu, long long int strideU, float *V, int ldv, long long int strideV,
    float *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, double *A, int lda, long long int strideA,
    double *S, long long int strideS, double *U, int ldu, long long int strideU, double *V, int ldv, long long int strideV,
    double *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDgesvdaStridedBatched(
    handle, jobz, rank, m, n, A, lda, strideA, S, strideS, U, ldu, strideU, V, ldv, strideV, work, lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<c10::complex<float>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<float> *A, int lda, long long int strideA,
    float *S, long long int strideS, c10::complex<float> *U, int ldu, long long int strideU,
    c10::complex<float> *V, int ldv, long long int strideV,
    c10::complex<float> *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgesvdaStridedBatched(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<cuComplex*>(U),
    ldu, strideU,
    reinterpret_cast<cuComplex*>(V),
    ldv, strideV,
    reinterpret_cast<cuComplex*>(work),
    lwork, info, h_R_nrmF, batchSize
  ));
}

template<>
void gesvdaStridedBatched<c10::complex<double>>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, c10::complex<double> *A, int lda, long long int strideA,
    double *S, long long int strideS, c10::complex<double> *U, int ldu, long long int strideU,
    c10::complex<double> *V, int ldv, long long int strideV,
    c10::complex<double> *work, int lwork, int *info, double *h_R_nrmF, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgesvdaStridedBatched(
    handle, jobz, rank, m, n,
    reinterpret_cast<cuDoubleComplex*>(A),
    lda, strideA, S, strideS,
    reinterpret_cast<cuDoubleComplex*>(U),
    ldu, strideU,
    reinterpret_cast<cuDoubleComplex*>(V),
    ldv, strideV,
    reinterpret_cast<cuDoubleComplex*>(work),
    lwork, info, h_R_nrmF, batchSize
  ));
}
#endif


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
void geqrf_bufferSize<float>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<double>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<cuComplex*>(A), lda, lwork));
}

template <>
void geqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
      handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, lwork));
}

template <>
void geqrf<float>(CUDASOLVER_GEQRF_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<double>(CUDASOLVER_GEQRF_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo));
}

template <>
void geqrf<c10::complex<float>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      reinterpret_cast<cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(work),
      lwork,
      devInfo));
}

template <>
void geqrf<c10::complex<double>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZgeqrf(
      handle,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      reinterpret_cast<cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      devInfo));
}

template<>
void potrs<float>(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo));
}

template<>
void potrs<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const c10::complex<float> *A, int lda, c10::complex<float> *B, int ldb, int *devInfo
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const cuComplex*>(A),
    lda,
    reinterpret_cast<cuComplex*>(B),
    ldb, devInfo));
}

template<>
void potrs<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const c10::complex<double> *A, int lda, c10::complex<double> *B, int ldb, int *devInfo
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrs(
    handle, uplo, n, nrhs,
    reinterpret_cast<const cuDoubleComplex*>(A),
    lda,
    reinterpret_cast<cuDoubleComplex*>(B),
    ldb, devInfo));
}

template<>
void potrsBatched<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float *Aarray[], int lda, float *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnSpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template<>
void potrsBatched<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double *Aarray[], int lda, double *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnDpotrsBatched(handle, uplo, n, nrhs, Aarray, lda, Barray, ldb, info, batchSize));
}

template<>
void potrsBatched<c10::complex<float>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, c10::complex<float> *Aarray[], int lda, c10::complex<float> *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnCpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<cuComplex**>(Aarray),
    lda,
    reinterpret_cast<cuComplex**>(Barray),
    ldb, info, batchSize));
}

template<>
void potrsBatched<c10::complex<double>>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, c10::complex<double> *Aarray[], int lda, c10::complex<double> *Barray[], int ldb, int *info, int batchSize
) {
  TORCH_CUSOLVER_CHECK(cusolverDnZpotrsBatched(
    handle, uplo, n, nrhs,
    reinterpret_cast<cuDoubleComplex**>(Aarray),
    lda,
    reinterpret_cast<cuDoubleComplex**>(Barray),
    ldb, info, batchSize));
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

template <>
void ormqr_bufferSize<float>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<double>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork));
}

template <>
void ormqr_bufferSize<c10::complex<float>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<const cuComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr_bufferSize<c10::complex<double>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZunmqr_bufferSize(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<const cuDoubleComplex*>(C), ldc,
      lwork));
}

template <>
void ormqr<float>(CUDASOLVER_ORMQR_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<double>(CUDASOLVER_ORMQR_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo));
}

template <>
void ormqr<c10::complex<float>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnCunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuComplex*>(A), lda,
      reinterpret_cast<const cuComplex*>(tau),
      reinterpret_cast<cuComplex*>(C), ldc,
      reinterpret_cast<cuComplex*>(work), lwork,
      devInfo));
}

template <>
void ormqr<c10::complex<double>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnZunmqr(
      handle, side, trans,
      m, n, k,
      reinterpret_cast<const cuDoubleComplex*>(A), lda,
      reinterpret_cast<const cuDoubleComplex*>(tau),
      reinterpret_cast<cuDoubleComplex*>(C), ldc,
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

template <>
void syevd_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

template <>
void syevd_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork));
}

template <>
void syevd_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex*>(A),
      lda,
      W,
      lwork));
}

template <>
void syevd_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      W,
      lwork));
}

template <>
void syevd<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}

template <>
void syevd<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(
      cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info));
}

template <>
void syevd<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuComplex*>(work),
      lwork,
      info));
}

template <>
void syevd<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevd(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      info));
}

template <>
void syevj_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}

template <>
void syevj_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params));
}

template <>
void syevj_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex*>(A),
      lda,
      W,
      lwork,
      params));
}

template <>
void syevj_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      W,
      lwork,
      params));
}

template <>
void syevj<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsyevj(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
}

template <>
void syevj<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsyevj(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params));
}

template <>
void syevj<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevj(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuComplex*>(work),
      lwork,
      info,
      params));
}

template <>
void syevj<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info,
    syevjInfo_t params) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevj(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      info,
      params));
}

template <>
void syevjBatched_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
void syevjBatched_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
      handle, jobz, uplo, n, A, lda, W, lwork, params, batchsize));
}

template <>
void syevjBatched_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<float>* A,
    int lda,
    const float* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuComplex*>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

template <>
void syevjBatched_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const c10::complex<double>* A,
    int lda,
    const double* W,
    int* lwork,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      W,
      lwork,
      params,
      batchsize));
}

template <>
void syevjBatched<float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float* A,
    int lda,
    float* W,
    float* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnSsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
void syevjBatched<double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double* A,
    int lda,
    double* W,
    double* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnDsyevjBatched(
      handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchsize));
}

template <>
void syevjBatched<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<float>* A,
    int lda,
    float* W,
    c10::complex<float>* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnCheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuComplex*>(work),
      lwork,
      info,
      params,
      batchsize));
}

template <>
void syevjBatched<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    c10::complex<double>* A,
    int lda,
    double* W,
    c10::complex<double>* work,
    int lwork,
    int* info,
    syevjInfo_t params,
    int batchsize) {
  TORCH_CUSOLVER_CHECK(cusolverDnZheevjBatched(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      reinterpret_cast<cuDoubleComplex*>(work),
      lwork,
      info,
      params,
      batchsize));
}

#ifdef USE_CUSOLVER_64_BIT

void xpotrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info));
}

template <>
void xgeqrf_bufferSize<float>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      CUDA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<const void*>(tau),
      CUDA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<double>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      CUDA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<const void*>(tau),
      CUDA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      CUDA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_C_32F,
      reinterpret_cast<const void*>(tau),
      CUDA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(
      handle,
      params,
      m,
      n,
      CUDA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_C_64F,
      reinterpret_cast<const void*>(tau),
      CUDA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeqrf<float>(CUDASOLVER_XGEQRF_ARGTYPES(float)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      CUDA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(tau),
      CUDA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<double>(CUDASOLVER_XGEQRF_ARGTYPES(double)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      CUDA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(tau),
      CUDA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<c10::complex<float>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<float>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      CUDA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_C_32F,
      reinterpret_cast<void*>(tau),
      CUDA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeqrf<c10::complex<double>>(CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<double>)) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeqrf(
      handle,
      params,
      m,
      n,
      CUDA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_C_64F,
      reinterpret_cast<void*>(tau),
      CUDA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const float* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<const void*>(W),
      CUDA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<const void*>(W),
      CUDA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const c10::complex<float>* A,
    int64_t lda,
    const float* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<const void*>(W),
      CUDA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const c10::complex<double>* A,
    int64_t lda,
    const double* W,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<const void*>(W),
      CUDA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xsyevd<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    float* A,
    int64_t lda,
    float* W,
    float* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    float* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(W),
      CUDA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    double* A,
    int64_t lda,
    double* W,
    double* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    double* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(W),
      CUDA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    c10::complex<float>* A,
    int64_t lda,
    float* W,
    c10::complex<float>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<float>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(W),
      CUDA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xsyevd<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    c10::complex<double>* A,
    int64_t lda,
    double* W,
    c10::complex<double>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<double>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevd(
      handle,
      params,
      jobz,
      uplo,
      n,
      CUDA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(W),
      CUDA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

// cuSOLVER Xgeev bindings (requires cuSOLVER >= 11.7.2, i.e. CUDA 12.8+)
#if defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702)

template <>
void xgeev_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    const float* A,
    int64_t lda,
    const float* W,
    const float* VL,
    int64_t ldvl,
    const float* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      CUDA_R_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<const void*>(W),
      CUDA_R_32F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      CUDA_R_32F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      CUDA_R_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    const double* A,
    int64_t lda,
    const double* W,
    const double* VL,
    int64_t ldvl,
    const double* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      CUDA_R_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<const void*>(W),
      CUDA_R_64F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      CUDA_R_64F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      CUDA_R_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}


template <>
void xgeev_bufferSize<c10::complex<float>>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    const c10::complex<float>* A,
    int64_t lda,
    const c10::complex<float>* W,
    const c10::complex<float>* VL,
    int64_t ldvl,
    const c10::complex<float>* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      CUDA_C_32F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_C_32F,
      reinterpret_cast<const void*>(W),
      CUDA_C_32F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      CUDA_C_32F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      CUDA_C_32F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev_bufferSize<c10::complex<double>>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    const c10::complex<double>* A,
    int64_t lda,
    const c10::complex<double>* W,
    const c10::complex<double>* VL,
    int64_t ldvl,
    const c10::complex<double>* VR,
    int64_t ldvr,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost) {
  TORCH_CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      handle, params, jobvl, jobvr, n,
      CUDA_C_64F,
      reinterpret_cast<const void*>(A),
      lda,
      CUDA_C_64F,
      reinterpret_cast<const void*>(W),
      CUDA_C_64F,
      reinterpret_cast<const void*>(VL),
      ldvl,
      CUDA_C_64F,
      reinterpret_cast<const void*>(VR),
      ldvr,
      CUDA_C_64F,
      workspaceInBytesOnDevice,
      workspaceInBytesOnHost));
}

template <>
void xgeev<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    float* A,
    int64_t lda,
    float* W,
    float* VL,
    int64_t ldvl,
    float* VR,
    int64_t ldvr,
    float* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    float* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_CUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      CUDA_R_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_32F,
      reinterpret_cast<void*>(W),
      CUDA_R_32F,
      reinterpret_cast<void*>(VL),
      ldvl,
      CUDA_R_32F,
      reinterpret_cast<void*>(VR),
      ldvr,
      CUDA_R_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}




template <>
void xgeev<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    double* A,
    int64_t lda,
    double* W,
    double* VL,
    int64_t ldvl,
    double* VR,
    int64_t ldvr,
    double* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    double* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_CUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      CUDA_R_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_R_64F,
      reinterpret_cast<void*>(W),
      CUDA_R_64F,
      reinterpret_cast<void*>(VL),
      ldvl,
      CUDA_R_64F,
      reinterpret_cast<void*>(VR),
      ldvr,
      CUDA_R_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));

}

template <>
void xgeev<c10::complex<float>>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    c10::complex<float>* A,
    int64_t lda,
    c10::complex<float>* W,
    c10::complex<float>* VL,
    int64_t ldvl,
    c10::complex<float>* VR,
    int64_t ldvr,
    c10::complex<float>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<float>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_CUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      CUDA_C_32F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_C_32F,
      reinterpret_cast<void*>(W),
      CUDA_C_32F,
      reinterpret_cast<void*>(VL),
      ldvl,
      CUDA_C_32F,
      reinterpret_cast<void*>(VR),
      ldvr,
      CUDA_C_32F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}

template <>
void xgeev<c10::complex<double>>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobvl,
    cusolverEigMode_t jobvr,
    int64_t n,
    c10::complex<double>* A,
    int64_t lda,
    c10::complex<double>* W,
    c10::complex<double>* VL,
    int64_t ldvl,
    c10::complex<double>* VR,
    int64_t ldvr,
    c10::complex<double>* bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    c10::complex<double>* bufferOnHost,
    size_t workspaceInBytesOnHost,
    int* info) {

  TORCH_CUSOLVER_CHECK(cusolverDnXgeev(
      handle,
      params,
      jobvl,
      jobvr,
      n,
      CUDA_C_64F,
      reinterpret_cast<void*>(A),
      lda,
      CUDA_C_64F,
      reinterpret_cast<void*>(W),
      CUDA_C_64F,
      reinterpret_cast<void*>(VL),
      ldvl,
      CUDA_C_64F,
      reinterpret_cast<void*>(VR),
      ldvr,
      CUDA_C_64F,
      reinterpret_cast<void*>(bufferOnDevice),
      workspaceInBytesOnDevice,
      reinterpret_cast<void*>(bufferOnHost),
      workspaceInBytesOnHost,
      info));
}


#endif // defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702)


#endif // USE_CUSOLVER_64_BIT

#ifdef USE_CUSOLVER_64_BIT_XSYEV_BATCHED

template <>
void xsyevBatched_bufferSize<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const float *A,
    int64_t lda,
    const float *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_R_32F,
       reinterpret_cast<const void*>(A),
       lda,
       CUDA_R_32F,
       reinterpret_cast<const void*>(W),
       CUDA_R_32F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const double *A,
    int64_t lda,
    const double *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_R_64F,
       reinterpret_cast<const void*>(A),
       lda,
       CUDA_R_64F,
       reinterpret_cast<const void*>(W),
       CUDA_R_64F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const c10::complex<float> *A,
    int64_t lda,
    const float *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_C_32F,
       reinterpret_cast<const void*>(A),
       lda,
       CUDA_R_32F,
       reinterpret_cast<const void*>(W),
       CUDA_C_32F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched_bufferSize<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t  jobz,
    cublasFillMode_t uplo,
    int64_t n,
    const c10::complex<double> *A,
    int64_t lda,
    const double *W,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_C_64F,
       reinterpret_cast<const void*>(A),
       lda,
       CUDA_R_64F,
       reinterpret_cast<const void*>(W),
       CUDA_C_64F,
       workspaceInBytesOnDevice,
       workspaceInBytesOnHost,
       batchSize));
}

template <>
void xsyevBatched<float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    float *A,
    int64_t lda,
    float *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_R_32F,
       reinterpret_cast<void*>(A),
       lda,
       CUDA_R_32F,
       reinterpret_cast<void*>(W),
       CUDA_R_32F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    double *A,
    int64_t lda,
    double *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_R_64F,
       reinterpret_cast<void*>(A),
       lda,
       CUDA_R_64F,
       reinterpret_cast<void*>(W),
       CUDA_R_64F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<c10::complex<float>, float>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    c10::complex<float> *A,
    int64_t lda,
    float *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_C_32F,
       reinterpret_cast<void*>(A),
       lda,
       CUDA_R_32F,
       reinterpret_cast<void*>(W),
       CUDA_C_32F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

template <>
void xsyevBatched<c10::complex<double>, double>(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    c10::complex<double> *A,
    int64_t lda,
    double *W,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info,
    int64_t batchSize) {
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevBatched(
       handle,
       params,
       jobz,
       uplo,
       n,
       CUDA_C_64F,
       reinterpret_cast<void*>(A),
       lda,
       CUDA_R_64F,
       reinterpret_cast<void*>(W),
       CUDA_C_64F,
       bufferOnDevice,
       workspaceInBytesOnDevice,
       bufferOnHost,
       workspaceInBytesOnHost,
       info,
       batchSize));
}

#endif // USE_CUSOLVER_64_BIT_XSYEV_BATCHED

} // namespace at::cuda::solver
