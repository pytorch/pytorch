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


#define CUDASOLVER_GESVD_ARGTYPES(Dtype, Vtype)  \
    cusolverDnHandle_t handle, char jobu, char jobvt, int m, int n, Dtype* A, int lda, Vtype* S, Dtype* U, int ldu, Dtype* VT, \
    int ldvt, int* devInfo

template<class Dtype, class Vtype>
void gesvd(CUDASOLVER_GESVD_ARGTYPES(Dtype, Vtype)) {
  TORCH_CHECK(false, "at::cuda::solver::gesvd: not implemented for ", typeid(Dtype).name());
}
template<>
void gesvd<float>(CUDASOLVER_GESVD_ARGTYPES(float, float));
template<>
void gesvd<double>(CUDASOLVER_GESVD_ARGTYPES(double, double));
template<>
void gesvd<c10::complex<float>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<float>, float));
template<>
void gesvd<c10::complex<double>>(CUDASOLVER_GESVD_ARGTYPES(c10::complex<double>, double));


} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
