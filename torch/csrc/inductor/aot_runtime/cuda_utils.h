#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#define ATOI_RUNTIME_CUDA_CHECK(EXPR)                      \
  do {                                                     \
    const cudaError_t code = EXPR;                         \
    const char* msg = cudaGetErrorString(code);            \
    if (code != cudaSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)
