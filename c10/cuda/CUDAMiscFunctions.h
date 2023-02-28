#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/cuda/CUDAMacros.h>

#include <mutex>

namespace c10 {
namespace cuda {
C10_CUDA_API const char* get_cuda_check_suffix() noexcept;
C10_CUDA_API std::mutex* getFreeMutex();
} // namespace cuda
} // namespace c10
