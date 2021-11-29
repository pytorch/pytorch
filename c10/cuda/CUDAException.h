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

// For CUDA Runtime API
#ifdef STRIP_ERROR_MESSAGES
#define C10_CUDA_CHECK(EXPR)                                     \
  do {                                                           \
    cudaError_t __err = EXPR;                                    \
    if (__err != cudaSuccess) {                                  \
      throw c10::CUDAError(                                      \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          TORCH_CHECK_MSG(false, ""));                           \
    }                                                            \
  } while (0)
#else
#define C10_CUDA_CHECK(EXPR)                                        \
  do {                                                              \
    cudaError_t __err = EXPR;                                       \
    if (__err != cudaSuccess) {                                     \
      auto error_unused C10_UNUSED = cudaGetLastError();            \
      auto _cuda_check_suffix = c10::cuda::get_cuda_check_suffix(); \
      throw c10::CUDAError(                                         \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},    \
          TORCH_CHECK_MSG(                                          \
              false,                                                \
              "",                                                   \
              "CUDA error: ",                                       \
              cudaGetErrorString(__err),                            \
              _cuda_check_suffix));                                 \
    }                                                               \
  } while (0)
#endif

#define C10_CUDA_CHECK_WARN(EXPR)                              \
  do {                                                         \
    cudaError_t __err = EXPR;                                  \
    if (__err != cudaSuccess) {                                \
      auto error_unused C10_UNUSED = cudaGetLastError();       \
      TORCH_WARN("CUDA warning: ", cudaGetErrorString(__err)); \
    }                                                          \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
