#pragma once

#include <ATen/cuda/CUDAContext.h>

#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 11000
// cuSOLVER version >= 11000 includes 64-bit API
#define USE_CUSOLVER_64_BIT
#endif

#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && CUSOLVER_VERSION >= 11701
// cuSOLVER version >= 11701 includes 64-bit API for batched syev
#define USE_CUSOLVER_64_BIT_XSYEV_BATCHED
#endif

#if defined(CUDART_VERSION) || defined(USE_ROCM)

namespace at {
namespace cuda {
namespace solver {

#define CUDASOLVER_GETRF_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, int m, int n, Dtype* dA, int ldda, int* ipiv, int* info

template<class Dtype>
void getrf(CUDASOLVER_GETRF_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::getrf: not implemented");
}
template<>
void getrf<float>(CUDASOLVER_GETRF_ARGTYPES(float));
template<>
void getrf<double>(CUDASOLVER_GETRF_ARGTYPES(double));
template<>
void getrf<c10::complex<double>>(CUDASOLVER_GETRF_ARGTYPES(c10::complex<double>));
template<>
void getrf<c10::complex<float>>(CUDASOLVER_GETRF_ARGTYPES(c10::complex<float>));


#define CUDASOLVER_GETRS_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, int n, int nrhs, Dtype* dA, int lda, int* ipiv, Dtype* ret, int ldb, int* info, cublasOperation_t trans

template<class Dtype>
void getrs(CUDASOLVER_GETRS_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::getrs: not implemented");
}
template<>
void getrs<float>(CUDASOLVER_GETRS_ARGTYPES(float));
template<>
void getrs<double>(CUDASOLVER_GETRS_ARGTYPES(double));
template<>
void getrs<c10::complex<double>>(CUDASOLVER_GETRS_ARGTYPES(c10::complex<double>));
template<>
void getrs<c10::complex<float>>(CUDASOLVER_GETRS_ARGTYPES(c10::complex<float>));

#define CUDASOLVER_SYTRF_BUFFER_ARGTYPES(Dtype) \
  cusolverDnHandle_t handle, int n, Dtype *A, int lda, int *lwork

template <class Dtype>
void sytrf_bufferSize(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),
      "at::cuda::solver::sytrf_bufferSize: not implemented");
}
template <>
void sytrf_bufferSize<float>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(float));
template <>
void sytrf_bufferSize<double>(CUDASOLVER_SYTRF_BUFFER_ARGTYPES(double));
template <>
void sytrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<double>));
template <>
void sytrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_SYTRF_BUFFER_ARGTYPES(c10::complex<float>));

#define CUDASOLVER_SYTRF_ARGTYPES(Dtype)                                      \
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, Dtype *A, int lda, \
      int *ipiv, Dtype *work, int lwork, int *devInfo

template <class Dtype>
void sytrf(CUDASOLVER_SYTRF_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),
      "at::cuda::solver::sytrf: not implemented");
}
template <>
void sytrf<float>(CUDASOLVER_SYTRF_ARGTYPES(float));
template <>
void sytrf<double>(CUDASOLVER_SYTRF_ARGTYPES(double));
template <>
void sytrf<c10::complex<double>>(
    CUDASOLVER_SYTRF_ARGTYPES(c10::complex<double>));
template <>
void sytrf<c10::complex<float>>(CUDASOLVER_SYTRF_ARGTYPES(c10::complex<float>));

#define CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()  \
    cusolverDnHandle_t handle, int m, int n, int *lwork

template<class Dtype>
void gesvd_buffersize(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES()) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvd_buffersize: not implemented");
}
template<>
void gesvd_buffersize<float>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES());
template<>
void gesvd_buffersize<double>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES());
template<>
void gesvd_buffersize<c10::complex<float>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES());
template<>
void gesvd_buffersize<c10::complex<double>>(CUDASOLVER_GESVD_BUFFERSIZE_ARGTYPES());


#define CUDASOLVER_GESVD_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, Dtype *A, int lda, \
    Vtype *S, Dtype *U, int ldu, Dtype *VT, int ldvt, Dtype *work, int lwork, Vtype *rwork, int *info

template<class Dtype, class Vtype>
void gesvd(CUDASOLVER_GESVD_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvd: not implemented");
}
template<>
void gesvd<float>(CUDASOLVER_GESVD_ARGTYPES(float, float));
template<>
void gesvd<double>(CUDASOLVER_GESVD_ARGTYPES(double, double));
template<>
void gesvd<c10::complex<float>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<float>, float));
template<>
void gesvd<c10::complex<double>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<double>, double));


#define CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, Dtype *A, int lda, Vtype *S, \
    Dtype *U, int ldu, Dtype *V, int ldv, int *lwork, gesvdjInfo_t params

template<class Dtype, class Vtype>
void gesvdj_buffersize(CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvdj_buffersize: not implemented");
}
template<>
void gesvdj_buffersize<float>(CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(float, float));
template<>
void gesvdj_buffersize<double>(CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(double, double));
template<>
void gesvdj_buffersize<c10::complex<float>>(CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdj_buffersize<c10::complex<double>>(CUDASOLVER_GESVDJ_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));


#define CUDASOLVER_GESVDJ_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, Dtype* A, int lda, Vtype* S, Dtype* U, \
    int ldu, Dtype* V, int ldv, Dtype* work, int lwork, int *info, gesvdjInfo_t params

template<class Dtype, class Vtype>
void gesvdj(CUDASOLVER_GESVDJ_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvdj: not implemented");
}
template<>
void gesvdj<float>(CUDASOLVER_GESVDJ_ARGTYPES(float, float));
template<>
void gesvdj<double>(CUDASOLVER_GESVDJ_ARGTYPES(double, double));
template<>
void gesvdj<c10::complex<float>>(CUDASOLVER_GESVDJ_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdj<c10::complex<double>>(CUDASOLVER_GESVDJ_ARGTYPES(c10::complex<double>, double));


#define CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, Dtype* A, int lda, Vtype* S, Dtype* U, \
    int ldu, Dtype *V, int ldv, int *info, gesvdjInfo_t params, int batchSize

template<class Dtype, class Vtype>
void gesvdjBatched(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvdj: not implemented");
}
template<>
void gesvdjBatched<float>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(float, float));
template<>
void gesvdjBatched<double>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(double, double));
template<>
void gesvdjBatched<c10::complex<float>>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdjBatched<c10::complex<double>>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, Dtype *A, int lda, long long int strideA, \
    Vtype *S, long long int strideS, Dtype *U, int ldu, long long int strideU, Dtype *V, int ldv, long long int strideV, \
    int *lwork, int batchSize

template<class Dtype, class Vtype>
void gesvdaStridedBatched_buffersize(CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvdaStridedBatched_buffersize: not implemented");
}
template<>
void gesvdaStridedBatched_buffersize<float>(CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(float, float));
template<>
void gesvdaStridedBatched_buffersize<double>(CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(double, double));
template<>
void gesvdaStridedBatched_buffersize<c10::complex<float>>(CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdaStridedBatched_buffersize<c10::complex<double>>(CUDASOLVER_GESVDA_STRIDED_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));


#define CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, Dtype *A, int lda, long long int strideA, \
    Vtype *S, long long int strideS, Dtype *U, int ldu, long long int strideU, Dtype *V, int ldv, long long int strideV, \
    Dtype *work, int lwork, int *info, double *h_R_nrmF, int batchSize
// h_R_nrmF is always double, regardless of input Dtype.

template<class Dtype, class Vtype>
void gesvdaStridedBatched(CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(Dtype, Vtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::gesvdaStridedBatched: not implemented");
}
template<>
void gesvdaStridedBatched<float>(CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(float, float));
template<>
void gesvdaStridedBatched<double>(CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(double, double));
template<>
void gesvdaStridedBatched<c10::complex<float>>(CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdaStridedBatched<c10::complex<double>>(CUDASOLVER_GESVDA_STRIDED_BATCHED_ARGTYPES(c10::complex<double>, double));


#define CUDASOLVER_POTRF_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, Dtype* A, int lda, Dtype* work, int lwork, int* info

template<class Dtype>
void potrf(CUDASOLVER_POTRF_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::potrf: not implemented");
}
template<>
void potrf<float>(CUDASOLVER_POTRF_ARGTYPES(float));
template<>
void potrf<double>(CUDASOLVER_POTRF_ARGTYPES(double));
template<>
void potrf<c10::complex<float>>(CUDASOLVER_POTRF_ARGTYPES(c10::complex<float>));
template<>
void potrf<c10::complex<double>>(CUDASOLVER_POTRF_ARGTYPES(c10::complex<double>));


#define CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, Dtype* A, int lda, int* lwork

template<class Dtype>
void potrf_buffersize(CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::potrf_buffersize: not implemented");
}
template<>
void potrf_buffersize<float>(CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(float));
template<>
void potrf_buffersize<double>(CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(double));
template<>
void potrf_buffersize<c10::complex<float>>(CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template<>
void potrf_buffersize<c10::complex<double>>(CUDASOLVER_POTRF_BUFFERSIZE_ARGTYPES(c10::complex<double>));


#define CUDASOLVER_POTRF_BATCHED_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, Dtype** A, int lda, int* info, int batchSize

template<class Dtype>
void potrfBatched(CUDASOLVER_POTRF_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::potrfBatched: not implemented");
}
template<>
void potrfBatched<float>(CUDASOLVER_POTRF_BATCHED_ARGTYPES(float));
template<>
void potrfBatched<double>(CUDASOLVER_POTRF_BATCHED_ARGTYPES(double));
template<>
void potrfBatched<c10::complex<float>>(CUDASOLVER_POTRF_BATCHED_ARGTYPES(c10::complex<float>));
template<>
void potrfBatched<c10::complex<double>>(CUDASOLVER_POTRF_BATCHED_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(scalar_t) \
  cusolverDnHandle_t handle, int m, int n, scalar_t *A, int lda, int *lwork

template <class scalar_t>
void geqrf_bufferSize(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::geqrf_bufferSize: not implemented");
}
template <>
void geqrf_bufferSize<float>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(float));
template <>
void geqrf_bufferSize<double>(CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(double));
template <>
void geqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void geqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_GEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_GEQRF_ARGTYPES(scalar_t)                      \
  cusolverDnHandle_t handle, int m, int n, scalar_t *A, int lda, \
      scalar_t *tau, scalar_t *work, int lwork, int *devInfo

template <class scalar_t>
void geqrf(CUDASOLVER_GEQRF_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::geqrf: not implemented");
}
template <>
void geqrf<float>(CUDASOLVER_GEQRF_ARGTYPES(float));
template <>
void geqrf<double>(CUDASOLVER_GEQRF_ARGTYPES(double));
template <>
void geqrf<c10::complex<float>>(CUDASOLVER_GEQRF_ARGTYPES(c10::complex<float>));
template <>
void geqrf<c10::complex<double>>(
    CUDASOLVER_GEQRF_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_POTRS_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const Dtype *A, int lda, Dtype *B, int ldb, int *devInfo

template<class Dtype>
void potrs(CUDASOLVER_POTRS_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::potrs: not implemented");
}
template<>
void potrs<float>(CUDASOLVER_POTRS_ARGTYPES(float));
template<>
void potrs<double>(CUDASOLVER_POTRS_ARGTYPES(double));
template<>
void potrs<c10::complex<float>>(CUDASOLVER_POTRS_ARGTYPES(c10::complex<float>));
template<>
void potrs<c10::complex<double>>(CUDASOLVER_POTRS_ARGTYPES(c10::complex<double>));


#define CUDASOLVER_POTRS_BATCHED_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, Dtype *Aarray[], int lda, Dtype *Barray[], int ldb, int *info, int batchSize

template<class Dtype>
void potrsBatched(CUDASOLVER_POTRS_BATCHED_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::potrsBatched: not implemented");
}
template<>
void potrsBatched<float>(CUDASOLVER_POTRS_BATCHED_ARGTYPES(float));
template<>
void potrsBatched<double>(CUDASOLVER_POTRS_BATCHED_ARGTYPES(double));
template<>
void potrsBatched<c10::complex<float>>(CUDASOLVER_POTRS_BATCHED_ARGTYPES(c10::complex<float>));
template<>
void potrsBatched<c10::complex<double>>(CUDASOLVER_POTRS_BATCHED_ARGTYPES(c10::complex<double>));


#define CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(Dtype)                        \
  cusolverDnHandle_t handle, int m, int n, int k, const Dtype *A, int lda, \
      const Dtype *tau, int *lwork

template <class Dtype>
void orgqr_buffersize(CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::orgqr_buffersize: not implemented");
}
template <>
void orgqr_buffersize<float>(CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(float));
template <>
void orgqr_buffersize<double>(CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(double));
template <>
void orgqr_buffersize<c10::complex<float>>(CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void orgqr_buffersize<c10::complex<double>>(CUDASOLVER_ORGQR_BUFFERSIZE_ARGTYPES(c10::complex<double>));


#define CUDASOLVER_ORGQR_ARGTYPES(Dtype)                             \
  cusolverDnHandle_t handle, int m, int n, int k, Dtype *A, int lda, \
      const Dtype *tau, Dtype *work, int lwork, int *devInfo

template <class Dtype>
void orgqr(CUDASOLVER_ORGQR_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype), "at::cuda::solver::orgqr: not implemented");
}
template <>
void orgqr<float>(CUDASOLVER_ORGQR_ARGTYPES(float));
template <>
void orgqr<double>(CUDASOLVER_ORGQR_ARGTYPES(double));
template <>
void orgqr<c10::complex<float>>(CUDASOLVER_ORGQR_ARGTYPES(c10::complex<float>));
template <>
void orgqr<c10::complex<double>>(CUDASOLVER_ORGQR_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(Dtype)                          \
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, \
      int m, int n, int k, const Dtype *A, int lda, const Dtype *tau,        \
      const Dtype *C, int ldc, int *lwork

template <class Dtype>
void ormqr_bufferSize(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),
      "at::cuda::solver::ormqr_bufferSize: not implemented");
}
template <>
void ormqr_bufferSize<float>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(float));
template <>
void ormqr_bufferSize<double>(CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(double));
template <>
void ormqr_bufferSize<c10::complex<float>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void ormqr_bufferSize<c10::complex<double>>(
    CUDASOLVER_ORMQR_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_ORMQR_ARGTYPES(Dtype)                                     \
  cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, \
      int m, int n, int k, const Dtype *A, int lda, const Dtype *tau, Dtype *C,    \
      int ldc, Dtype *work, int lwork, int *devInfo

template <class Dtype>
void ormqr(CUDASOLVER_ORMQR_ARGTYPES(Dtype)) {
  static_assert(false&&sizeof(Dtype),
      "at::cuda::solver::ormqr: not implemented");
}
template <>
void ormqr<float>(CUDASOLVER_ORMQR_ARGTYPES(float));
template <>
void ormqr<double>(CUDASOLVER_ORMQR_ARGTYPES(double));
template <>
void ormqr<c10::complex<float>>(CUDASOLVER_ORMQR_ARGTYPES(c10::complex<float>));
template <>
void ormqr<c10::complex<double>>(
    CUDASOLVER_ORMQR_ARGTYPES(c10::complex<double>));

#ifdef USE_CUSOLVER_64_BIT

template<class Dtype>
cudaDataType get_cusolver_datatype() {
  static_assert(false&&sizeof(Dtype), "cusolver doesn't support data type");
  return {};
}
template<> cudaDataType get_cusolver_datatype<float>();
template<> cudaDataType get_cusolver_datatype<double>();
template<> cudaDataType get_cusolver_datatype<c10::complex<float>>();
template<> cudaDataType get_cusolver_datatype<c10::complex<double>>();

void xpotrf_buffersize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost);

void xpotrf(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A,
    int64_t lda, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost,
    int *info);

void xpotrs(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeB, void *B, int64_t ldb, int *info);

#endif // USE_CUSOLVER_64_BIT

#define CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(scalar_t, value_t)             \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, const scalar_t *A, int lda, const value_t *W, int *lwork

template <class scalar_t, class value_t = scalar_t>
void syevd_bufferSize(CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::syevd_bufferSize: not implemented");
}

template <>
void syevd_bufferSize<float>(
    CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(float, float));
template <>
void syevd_bufferSize<double>(
    CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(double, double));
template <>
void syevd_bufferSize<c10::complex<float>, float>(
    CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template <>
void syevd_bufferSize<c10::complex<double>, double>(
    CUDASOLVER_SYEVD_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_SYEVD_ARGTYPES(scalar_t, value_t)                        \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, scalar_t *A, int lda, value_t *W, scalar_t *work, int lwork,   \
      int *info

template <class scalar_t, class value_t = scalar_t>
void syevd(CUDASOLVER_SYEVD_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::syevd: not implemented");
}

template <>
void syevd<float>(CUDASOLVER_SYEVD_ARGTYPES(float, float));
template <>
void syevd<double>(CUDASOLVER_SYEVD_ARGTYPES(double, double));
template <>
void syevd<c10::complex<float>, float>(
    CUDASOLVER_SYEVD_ARGTYPES(c10::complex<float>, float));
template <>
void syevd<c10::complex<double>, double>(
    CUDASOLVER_SYEVD_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(scalar_t, value_t)             \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, const scalar_t *A, int lda, const value_t *W, int *lwork,      \
      syevjInfo_t params

template <class scalar_t, class value_t = scalar_t>
void syevj_bufferSize(CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::syevj_bufferSize: not implemented");
}

template <>
void syevj_bufferSize<float>(
    CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(float, float));
template <>
void syevj_bufferSize<double>(
    CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(double, double));
template <>
void syevj_bufferSize<c10::complex<float>, float>(
    CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template <>
void syevj_bufferSize<c10::complex<double>, double>(
    CUDASOLVER_SYEVJ_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_SYEVJ_ARGTYPES(scalar_t, value_t)                        \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, scalar_t *A, int lda, value_t *W, scalar_t *work, int lwork,   \
      int *info, syevjInfo_t params

template <class scalar_t, class value_t = scalar_t>
void syevj(CUDASOLVER_SYEVJ_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t), "at::cuda::solver::syevj: not implemented");
}

template <>
void syevj<float>(CUDASOLVER_SYEVJ_ARGTYPES(float, float));
template <>
void syevj<double>(CUDASOLVER_SYEVJ_ARGTYPES(double, double));
template <>
void syevj<c10::complex<float>, float>(
    CUDASOLVER_SYEVJ_ARGTYPES(c10::complex<float>, float));
template <>
void syevj<c10::complex<double>, double>(
    CUDASOLVER_SYEVJ_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t)     \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, const scalar_t *A, int lda, const value_t *W, int *lwork,      \
      syevjInfo_t params, int batchsize

template <class scalar_t, class value_t = scalar_t>
void syevjBatched_bufferSize(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::syevjBatched_bufferSize: not implemented");
}

template <>
void syevjBatched_bufferSize<float>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(float, float));
template <>
void syevjBatched_bufferSize<double>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(double, double));
template <>
void syevjBatched_bufferSize<c10::complex<float>, float>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template <>
void syevjBatched_bufferSize<c10::complex<double>, double>(
    CUDASOLVER_SYEVJ_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)                \
  cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, \
      int n, scalar_t *A, int lda, value_t *W, scalar_t *work, int lwork,   \
      int *info, syevjInfo_t params, int batchsize

template <class scalar_t, class value_t = scalar_t>
void syevjBatched(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::syevjBatched: not implemented");
}

template <>
void syevjBatched<float>(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(float, float));
template <>
void syevjBatched<double>(CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(double, double));
template <>
void syevjBatched<c10::complex<float>, float>(
    CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(c10::complex<float>, float));
template <>
void syevjBatched<c10::complex<double>, double>(
    CUDASOLVER_SYEVJ_BATCHED_ARGTYPES(c10::complex<double>, double));

#ifdef USE_CUSOLVER_64_BIT

#define CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(scalar_t)                       \
  cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, \
      const scalar_t *A, int64_t lda, const scalar_t *tau,                    \
      size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost

template <class scalar_t>
void xgeqrf_bufferSize(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xgeqrf_bufferSize: not implemented");
}

template <>
void xgeqrf_bufferSize<float>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(float));
template <>
void xgeqrf_bufferSize<double>(CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(double));
template <>
void xgeqrf_bufferSize<c10::complex<float>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void xgeqrf_bufferSize<c10::complex<double>>(
    CUDASOLVER_XGEQRF_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_XGEQRF_ARGTYPES(scalar_t)                                  \
  cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, \
      scalar_t *A, int64_t lda, scalar_t *tau, scalar_t *bufferOnDevice,      \
      size_t workspaceInBytesOnDevice, scalar_t *bufferOnHost,                \
      size_t workspaceInBytesOnHost, int *info

template <class scalar_t>
void xgeqrf(CUDASOLVER_XGEQRF_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t), "at::cuda::solver::xgeqrf: not implemented");
}

template <>
void xgeqrf<float>(CUDASOLVER_XGEQRF_ARGTYPES(float));
template <>
void xgeqrf<double>(CUDASOLVER_XGEQRF_ARGTYPES(double));
template <>
void xgeqrf<c10::complex<float>>(
    CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<float>));
template <>
void xgeqrf<c10::complex<double>>(
    CUDASOLVER_XGEQRF_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(scalar_t, value_t) \
  cusolverDnHandle_t handle, cusolverDnParams_t params,          \
      cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,  \
      const scalar_t *A, int64_t lda, const value_t *W,          \
      size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost

template <class scalar_t, class value_t = scalar_t>
void xsyevd_bufferSize(
    CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xsyevd_bufferSize: not implemented");
}

template <>
void xsyevd_bufferSize<float>(
    CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(float, float));
template <>
void xsyevd_bufferSize<double>(
    CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(double, double));
template <>
void xsyevd_bufferSize<c10::complex<float>, float>(
    CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));
template <>
void xsyevd_bufferSize<c10::complex<double>, double>(
    CUDASOLVER_XSYEVD_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_XSYEVD_ARGTYPES(scalar_t, value_t)                        \
  cusolverDnHandle_t handle, cusolverDnParams_t params,                      \
      cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n, scalar_t *A, \
      int64_t lda, value_t *W, scalar_t *bufferOnDevice,                     \
      size_t workspaceInBytesOnDevice, scalar_t *bufferOnHost,               \
      size_t workspaceInBytesOnHost, int *info

template <class scalar_t, class value_t = scalar_t>
void xsyevd(CUDASOLVER_XSYEVD_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xsyevd: not implemented");
}

template <>
void xsyevd<float>(CUDASOLVER_XSYEVD_ARGTYPES(float, float));
template <>
void xsyevd<double>(CUDASOLVER_XSYEVD_ARGTYPES(double, double));
template <>
void xsyevd<c10::complex<float>, float>(
    CUDASOLVER_XSYEVD_ARGTYPES(c10::complex<float>, float));
template <>
void xsyevd<c10::complex<double>, double>(
    CUDASOLVER_XSYEVD_ARGTYPES(c10::complex<double>, double));



// cuSOLVER Xgeev (non-Hermitian eigen decomposition, CUDA >= 12.8)
#if defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702)

#define CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(scalar_t)                        \
cusolverDnHandle_t handle, cusolverDnParams_t params,                         \
cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n,                  \
const scalar_t* A, int64_t lda, const scalar_t* W,                            \
const scalar_t* VL, int64_t ldvl, const scalar_t* VR, int64_t ldvr,           \
size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost

template <class scalar_t>
void xgeev_bufferSize(
    CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xgeev_bufferSize: not implemented");
}

template <>
void xgeev_bufferSize<float>(CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(float));

template <>
void xgeev_bufferSize<double>(CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(double));

template <>
void xgeev_bufferSize<c10::complex<float>>(
    CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(c10::complex<float>));

template <>
void xgeev_bufferSize<c10::complex<double>>(
    CUDASOLVER_XGEEV_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUDASOLVER_XGEEV_ARGTYPES(scalar_t)                                    \
cusolverDnHandle_t handle, cusolverDnParams_t params,                          \
cusolverEigMode_t jobvl, cusolverEigMode_t jobvr, int64_t n, scalar_t *A,      \
int64_t lda, scalar_t *W, scalar_t *VL, int64_t ldvl, scalar_t *VR, int64_t ldvr,\
scalar_t *bufferOnDevice, size_t workspaceInBytesOnDevice, scalar_t *bufferOnHost,\
size_t workspaceInBytesOnHost, int *info

template <class scalar_t>
void xgeev(CUDASOLVER_XGEEV_ARGTYPES(scalar_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xgeev: not implemented");
}

template <>
void xgeev<float>(CUDASOLVER_XGEEV_ARGTYPES(float));

template <>
void xgeev<double>(CUDASOLVER_XGEEV_ARGTYPES(double));

template <>
void xgeev<c10::complex<float>>(CUDASOLVER_XGEEV_ARGTYPES(c10::complex<float>));

template <>
void xgeev<c10::complex<double>>(CUDASOLVER_XGEEV_ARGTYPES(c10::complex<double>));

#endif // defined(CUSOLVER_VERSION) && (CUSOLVER_VERSION >= 11702)

#endif // USE_CUSOLVER_64_BIT

#ifdef USE_CUSOLVER_64_BIT_XSYEV_BATCHED

#define CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t) \
    cusolverDnHandle_t handle,                                          \
    cusolverDnParams_t params,                                          \
    cusolverEigMode_t  jobz,                                            \
    cublasFillMode_t uplo,                                              \
    int64_t n,                                                          \
    const scalar_t *A,                                                  \
    int64_t lda,                                                        \
    const value_t *W,                                                   \
    size_t *workspaceInBytesOnDevice,                                   \
    size_t *workspaceInBytesOnHost,                                     \
    int64_t batchSize

template <class scalar_t, class value_t = scalar_t>
void xsyevBatched_bufferSize(
    CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xsyevBatched_bufferSize: not implemented");
}

template <>
void xsyevBatched_bufferSize<float>(
    CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(float, float));

template <>
void xsyevBatched_bufferSize<double>(
    CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(double, double));

template <>
void xsyevBatched_bufferSize<c10::complex<float>, float>(
    CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<float>, float));

template <>
void xsyevBatched_bufferSize<c10::complex<double>, double>(
    CUDASOLVER_XSYEV_BATCHED_BUFFERSIZE_ARGTYPES(c10::complex<double>, double));

#define CUDASOLVER_XSYEV_BATCHED_ARGTYPES(scalar_t, value_t) \
    cusolverDnHandle_t handle,                               \
    cusolverDnParams_t params,                               \
    cusolverEigMode_t jobz,                                  \
    cublasFillMode_t uplo,                                   \
    int64_t n,                                               \
    scalar_t *A,                                             \
    int64_t lda,                                             \
    value_t *W,                                              \
    void *bufferOnDevice,                                    \
    size_t workspaceInBytesOnDevice,                         \
    void *bufferOnHost,                                      \
    size_t workspaceInBytesOnHost,                           \
    int *info,                                               \
    int64_t batchSize

template <class scalar_t, class value_t = scalar_t>
void xsyevBatched(CUDASOLVER_XSYEV_BATCHED_ARGTYPES(scalar_t, value_t)) {
  static_assert(false&&sizeof(scalar_t),
      "at::cuda::solver::xsyevBatched: not implemented");
}

template <>
void xsyevBatched<float>(
    CUDASOLVER_XSYEV_BATCHED_ARGTYPES(float, float));

template <>
void xsyevBatched<double>(
    CUDASOLVER_XSYEV_BATCHED_ARGTYPES(double, double));

template <>
void xsyevBatched<c10::complex<float>, float>(
    CUDASOLVER_XSYEV_BATCHED_ARGTYPES(c10::complex<float>, float));

template <>
void xsyevBatched<c10::complex<double>, double>(
    CUDASOLVER_XSYEV_BATCHED_ARGTYPES(c10::complex<double>, double));

#endif // USE_CUSOLVER_64_BIT_XSYEV_BATCHED

} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
