//NS: CUDACachingAllocator must be included before to get CUDART_VERSION definedi
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/cuda/Exceptions.h>

namespace at::cuda {
namespace blas {

C10_EXPORT const char* _cublasGetErrorEnum(cublasStatus_t error) {
  if (error == CUBLAS_STATUS_SUCCESS) {
    return "CUBLAS_STATUS_SUCCESS";
  }
  if (error == CUBLAS_STATUS_NOT_INITIALIZED) {
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  }
  if (error == CUBLAS_STATUS_ALLOC_FAILED) {
    return "CUBLAS_STATUS_ALLOC_FAILED";
  }
  if (error == CUBLAS_STATUS_INVALID_VALUE) {
    return "CUBLAS_STATUS_INVALID_VALUE";
  }
  if (error == CUBLAS_STATUS_ARCH_MISMATCH) {
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  }
  if (error == CUBLAS_STATUS_MAPPING_ERROR) {
    return "CUBLAS_STATUS_MAPPING_ERROR";
  }
  if (error == CUBLAS_STATUS_EXECUTION_FAILED) {
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  }
  if (error == CUBLAS_STATUS_INTERNAL_ERROR) {
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  if (error == CUBLAS_STATUS_NOT_SUPPORTED) {
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  }
#ifdef CUBLAS_STATUS_LICENSE_ERROR
  if (error == CUBLAS_STATUS_LICENSE_ERROR) {
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
#endif
  return "<unknown>";
}

} // namespace blas

namespace solver {
#if !defined(USE_ROCM)

C10_EXPORT const char* cusolverGetErrorMessage(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:                     return "CUSOLVER_STATUS_SUCCESS";
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

#else

C10_EXPORT const char* hipsolverGetErrorMessage(hipsolverStatus_t status) {
  switch (status) {
    case HIPSOLVER_STATUS_SUCCESS:                    return "HIPSOLVER_STATUS_SUCCESS";
    case HIPSOLVER_STATUS_NOT_INITIALIZED:            return "HIPSOLVER_STATUS_NOT_INITIALIZED";
    case HIPSOLVER_STATUS_ALLOC_FAILED:               return "HIPSOLVER_STATUS_ALLOC_FAILED";
    case HIPSOLVER_STATUS_INVALID_VALUE:              return "HIPSOLVER_STATUS_INVALID_VALUE";
    case HIPSOLVER_STATUS_MAPPING_ERROR:              return "HIPSOLVER_STATUS_MAPPING_ERROR";
    case HIPSOLVER_STATUS_EXECUTION_FAILED:           return "HIPSOLVER_STATUS_EXECUTION_FAILED";
    case HIPSOLVER_STATUS_INTERNAL_ERROR:             return "HIPSOLVER_STATUS_INTERNAL_ERROR";
    case HIPSOLVER_STATUS_NOT_SUPPORTED:              return "HIPSOLVER_STATUS_NOT_SUPPORTED";
    case HIPSOLVER_STATUS_ARCH_MISMATCH:              return "HIPSOLVER_STATUS_ARCH_MISMATCH";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:          return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
    case HIPSOLVER_STATUS_INVALID_ENUM:               return "HIPSOLVER_STATUS_INVALID_ENUM";
    case HIPSOLVER_STATUS_UNKNOWN:                    return "HIPSOLVER_STATUS_UNKNOWN";
    case HIPSOLVER_STATUS_ZERO_PIVOT:                 return "HIPSOLVER_STATUS_ZERO_PIVOT";
    default:                                          return "Unknown hipsolver error number";
  }
}

#endif
} // namespace solver

#if defined(USE_CUDSS)
namespace cudss {

C10_EXPORT const char* cudssGetErrorMessage(cudssStatus_t status) {
  switch (status) {
    case CUDSS_STATUS_SUCCESS:                     return "CUDSS_STATUS_SUCCESS";
    case CUDSS_STATUS_NOT_INITIALIZED:             return "CUDSS_STATUS_NOT_INITIALIZED";
    case CUDSS_STATUS_ALLOC_FAILED:                return "CUDSS_STATUS_ALLOC_FAILED";
    case CUDSS_STATUS_INVALID_VALUE:               return "CUDSS_STATUS_INVALID_VALUE";
    case CUDSS_STATUS_NOT_SUPPORTED:               return "CUDSS_STATUS_NOT_SUPPORTED";
    case CUDSS_STATUS_EXECUTION_FAILED:            return "CUDSS_STATUS_EXECUTION_FAILED";
    case CUDSS_STATUS_INTERNAL_ERROR:              return "CUDSS_STATUS_INTERNAL_ERROR";
    default:                                       return "Unknown cudss error number";
  }
}

} // namespace cudss
#endif

} // namespace at::cuda
