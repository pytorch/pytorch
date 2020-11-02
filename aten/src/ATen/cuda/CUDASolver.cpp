#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDASolver.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifdef CUDART_VERSION

namespace at {
namespace cuda {
namespace solver {

template <>
void getrf<double>(
    cusolverDnHandle_t handle, int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  void* buffer = allocator.allocate(sizeof(double)*lwork).get();
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(
      handle, m, n, dA, ldda, static_cast<double*>(buffer), ipiv, info));
}

template <>
void getrf<float>(
    cusolverDnHandle_t handle, int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  int lwork;
  TORCH_CUSOLVER_CHECK(
      cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  void* buffer = allocator.allocate(sizeof(float)*lwork).get();
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(
      handle, m, n, dA, ldda, static_cast<float*>(buffer), ipiv, info));
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

} // namespace solver
} // namespace cuda
} // namespace at

#endif // CUDART_VERSION
