#pragma once

#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAMiscFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

// Used to denote errors from CUDA framework.
// This needs to be declared here instead util/Exception.h for proper conversion
// during hipify.
namespace c10 {
class C10_CUDA_API CUDAError : public c10::Error {
  using Error::Error;
};
} // namespace c10

#define C10_CUDA_CHECK(EXPR)                                               \
  do {                                                                     \
    const cudaError_t __err = EXPR;                                        \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                              \
      c10::cuda::c10_cuda_check_implementation(                            \
          __FILE__,                                                        \
          __func__, /* Line number's data type is not well-defined between \
                       compilers, so we perform an explicit cast */        \
          static_cast<uint32_t>(__LINE__),                                 \
          true);                                                           \
    }                                                                      \
  } while (0)

#define C10_CUDA_CHECK_WARN(EXPR)                              \
  do {                                                         \
    const cudaError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                  \
      auto error_unused C10_UNUSED = cudaGetLastError();       \
      (void)error_unused;                                      \
      TORCH_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
    }                                                          \
  } while (0)

// Indicates that a CUDA error is handled in a non-standard way
#define C10_CUDA_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a CUDA error
#define C10_CUDA_IGNORE_ERROR(EXPR)                             \
  do {                                                          \
    const cudaError_t __err = EXPR;                             \
    if (C10_UNLIKELY(__err != cudaSuccess)) {                   \
      cudaError_t error_unused C10_UNUSED = cudaGetLastError(); \
      (void)error_unused;                                       \
    }                                                           \
  } while (0)

// Clear the last CUDA error
#define C10_CUDA_CLEAR_ERROR()                                \
  do {                                                        \
    cudaError_t error_unused C10_UNUSED = cudaGetLastError(); \
    (void)error_unused;                                       \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

namespace c10 {
namespace cuda {

/// In the event of a CUDA failure, formats a nice error message about that
/// failure and also checks for device-side assertion failures
C10_CUDA_API void c10_cuda_check_implementation(
    const std::string& filename,
    const std::string& function_name,
    const int line_number,
    const bool include_device_assertions);

} // namespace cuda
} // namespace c10
