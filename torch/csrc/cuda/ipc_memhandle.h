#pragma once

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <THC/THC.h>


namespace torch {
  void* getCachedCUDAIpcDevptr(std::string handle);
} // namespace torch
#endif
