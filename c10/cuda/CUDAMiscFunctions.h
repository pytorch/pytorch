#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/cuda/CUDAMacros.h>
#include <cuda_runtime.h>

#include <mutex>
#include <string>

namespace c10::cuda {
C10_CUDA_API std::string get_cuda_error_help(cudaError_t) noexcept;
C10_CUDA_API const char* get_cuda_check_suffix() noexcept;
C10_CUDA_API std::mutex* getFreeMutex();
} // namespace c10::cuda
