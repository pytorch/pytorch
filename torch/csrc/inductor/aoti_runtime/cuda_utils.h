#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <cuda.h>
#include <cuda_runtime_api.h>

#define AOTI_RUNTIME_CUDA_CHECK(EXPR)                      \
  do {                                                     \
    const cudaError_t code = EXPR;                         \
    const char* msg = cudaGetErrorString(code);            \
    if (code != cudaSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)
