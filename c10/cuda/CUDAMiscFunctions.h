#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/cuda/CUDAMacros.h>

namespace c10 {
namespace cuda {
C10_CUDA_API std::string get_cuda_check_suffix(cudaError_t err) noexcept;
}
} // namespace c10
