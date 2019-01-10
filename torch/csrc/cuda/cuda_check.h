#pragma once

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

namespace torch {
// We're using three CUDA APIs, so define a few helpers for error handling
static inline void nvrtcCheck(nvrtcResult result,const char * file, int line) {
  if(result != NVRTC_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << nvrtcGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_NVRTC_CHECK(result) ::torch::nvrtcCheck(result,__FILE__,__LINE__);

static inline void cuCheck(CUresult result, const char * file, int line) {
  if(result != CUDA_SUCCESS) {
    const char * str;
    cuGetErrorString(result, &str);
    std::stringstream ss;
    ss << file << ":" << line << ": " << str;
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CU_CHECK(result) ::torch::cuCheck(result,__FILE__,__LINE__);

static inline void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CUDA_CHECK(result) ::torch::cudaCheck(result,__FILE__,__LINE__);

}

#endif
