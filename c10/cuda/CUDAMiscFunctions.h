#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime.h>

#include <mutex>
#include <string>
#include <string_view>

namespace c10::cuda {
C10_CUDA_API std::string get_cuda_error_help(cudaError_t /*error*/) noexcept;
C10_CUDA_API const char* get_cuda_check_suffix() noexcept;
C10_CUDA_API std::mutex* getFreeMutex();
// Returns true if the error message indicates a sticky GPU error (one that
// requires cudaDeviceReset to recover from). A "CUDA error" or "HIP error"
// that is NOT an "invalid argument" error is considered sticky.
C10_CUDA_API bool isStickyGpuError(std::string_view msg) noexcept;
} // namespace c10::cuda
