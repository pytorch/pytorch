#pragma once
// this file is to avoid circular dependency between CUDAFunctions.h and
// CUDAExceptions.h

#include <c10/cuda/CUDAMacros.h>

namespace c10 {
namespace cuda {
C10_CUDA_API const char* get_cuda_check_suffix() noexcept;
}
} // namespace c10
