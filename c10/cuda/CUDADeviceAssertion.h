#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>

namespace c10::cuda {

#define CUDA_KERNEL_ASSERT2(condition) assert(condition)

} // namespace c10::cuda
