#pragma once

#include <ATen/cuda/CUDAContext.h>

#ifdef CUDART_VERSION

namespace at {
namespace cuda {
namespace solver {

#define CUDASOLVER_GETRF_ARGTYPES(Dtype)  \
    cusolverDnHandle_t handle, int m, int n, Dtype* dA, int ldda, int* ipiv, int* info

template<class Dtype>
void getrf(CUDASOLVER_GETRF_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::solver::getrf: not implemented for ", typeid(Dtype).name());
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
    cusolverDnHandle_t handle, int n, int nrhs, Dtype* dA, int lda, int* ipiv, Dtype* ret, int ldb, int* info

template<class Dtype>
void getrs(CUDASOLVER_GETRS_ARGTYPES(Dtype)) {
  TORCH_CHECK(false, "at::cuda::solver::getrs: not implemented for ", typeid(Dtype).name());
}
template<>
void getrs<float>(CUDASOLVER_GETRS_ARGTYPES(float));
template<>
void getrs<double>(CUDASOLVER_GETRS_ARGTYPES(double));
template<>
void getrs<c10::complex<double>>(CUDASOLVER_GETRS_ARGTYPES(c10::complex<double>));
template<>
void getrs<c10::complex<float>>(CUDASOLVER_GETRS_ARGTYPES(c10::complex<float>));


#define CUDASOLVER_GESVDJ_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, Dtype* A, int lda, Vtype* S, Dtype* U, \
    int ldu, Dtype *V, int ldv, int *info, gesvdjInfo_t params

template<class Dtype, class Vtype>
void gesvdj(CUDASOLVER_GESVDJ_ARGTYPES(Dtype, Vtype)) {
  TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::gesvdj: not implemented for ", typeid(Dtype).name());
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
  TORCH_INTERNAL_ASSERT(false, "at::cuda::solver::gesvdj: not implemented for ", typeid(Dtype).name());
}
template<>
void gesvdjBatched<float>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(float, float));
template<>
void gesvdjBatched<double>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(double, double));
template<>
void gesvdjBatched<c10::complex<float>>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(c10::complex<float>, float));
template<>
void gesvdjBatched<c10::complex<double>>(CUDASOLVER_GESVDJ_BATCHED_ARGTYPES(c10::complex<double>, double));

} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
