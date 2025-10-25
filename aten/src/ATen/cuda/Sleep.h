#pragma once
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::cuda {

// enqueues a kernel that spins for the specified number of cycles
TORCH_CUDA_CU_API void sleep(int64_t cycles);

// enqueues a kernel that spins until a flag is cleared by a
// corresponding call to clear_flag()
TORCH_CUDA_CU_API void busy_wait_for_flag();
TORCH_CUDA_CU_API void clear_flag();

// flushes instruction cache for ROCm; no-op for CUDA
TORCH_CUDA_CU_API void flush_icache();

}  // namespace at::cuda
