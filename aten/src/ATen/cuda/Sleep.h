#pragma once
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::cuda {

// enqueues a kernel that spins for the specified number of cycles
TORCH_CUDA_CU_API void sleep(int64_t cycles);

}  // namespace at::cuda
