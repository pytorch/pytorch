#include <torch/csrc/stable/c/shim.h>

#include <stdexcept>
#include <string>

/**
 * @brief Stable equivalent of C10_CUDA_CHECK().
 *
 * This macro wraps CUDA API calls and checks the returned cudaError_t. If an
 * error occurred, it generates a detailed error message using PyTorch's error
 * formatting and throws a std::runtime_error.
 *
 * @param EXPR A CUDA API call that returns a cudaError_t.
 *
 * @throws std::runtime_error If the CUDA call returns an error.
 *
 * @note Users of this macro are expected to include cuda_runtime.h.
 * @note Minimum compatible version: PyTorch 2.10.
 *
 * Example usage:
 * @code
 * STD_CUDA_CHECK(cudaMalloc(&ptr, size));
 * STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * @endcode
 */
#define STD_CUDA_CHECK(EXPR)                      \
  do {                                            \
    const cudaError_t __err = EXPR;               \
    char* __error_msg = nullptr;                  \
    torch_c10_cuda_check_msg(                     \
        static_cast<int32_t>(__err),              \
        __FILE__,                                 \
        __func__,                                 \
        static_cast<uint32_t>(__LINE__),          \
        true,                                     \
        &__error_msg);                            \
    if (__error_msg != nullptr) {                 \
      std::string __msg(__error_msg);             \
      torch_c10_cuda_free_error_msg(__error_msg); \
      throw std::runtime_error(__msg);            \
    }                                             \
  } while (0)

/**
 * @brief Stable equivalent of C10_CUDA_KERNEL_LAUNCH_CHECK().
 *
 * This macro should be called after a CUDA kernel launch to check for any
 * errors that occurred during the launch. It is equivalent to calling
 * `STD_CUDA_CHECK(cudaGetLastError())`.
 *
 * @throws std::runtime_error If the previous CUDA kernel launch had an error.
 *
 * @note Users of this macro are expected to include cuda_runtime.h.
 * @note Minimum compatible version: PyTorch 2.10.
 *
 * Example usage:
 * @code
 * my_kernel<<<blocks, threads, 0, stream>>>(args...);
 * STD_CUDA_KERNEL_LAUNCH_CHECK();
 * @endcode
 */
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(cudaGetLastError())
