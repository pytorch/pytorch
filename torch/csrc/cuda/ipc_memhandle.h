#pragma once

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>


namespace torch {
    std::shared_ptr<void> getCachedCUDAIpcDevptr(std::string handle);
} // namespace torch
#endif
