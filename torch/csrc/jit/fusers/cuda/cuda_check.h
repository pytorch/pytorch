#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "nvrtc.h"

#include <sstream>
#include <string>
#include <cstdio>
#include <stdexcept>

namespace torch { namespace jit { namespace cudafuser {

static inline void cudaReport(
  const char* name
, int error
, const char* file
, int line
, const char* error_string
, bool shouldError) {
  if (shouldError) {
    std::stringstream ss;
    ss << "CUDA FUSER " << name << " WARNING! file=" << file << " line=" << line << " error=" << error << " " << error_string << "\n";
    throw std::runtime_error(ss.str());
  } else {
    fprintf(stderr, "CUDA FUSER %s WARNING! file=%s line=%i error=%i %s\n"
      , name
      , file
      , line
      , error
      , error_string);
  }
}

static inline void nvrtcCheck(
  nvrtcResult result
, const char* file
, int line
, bool shouldError) {
  if (result != NVRTC_SUCCESS) 
    cudaReport("nvrtc", result, file, line, nvrtcGetErrorString(result), shouldError);
}

static inline void cuCheck(
  CUresult result
, const char* file
, int line
, bool shouldError) {
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    cudaReport("cu", result, file, line, error_string, shouldError);
  }
}

static inline void cudaCheck(
  cudaError_t result
, const char* file
, int line
, bool shouldError) {
  if (result != cudaSuccess)
    cudaReport("cuda", result, file, line, cudaGetErrorString(result), shouldError);
}

#define NVRTC_WARN(result) nvrtcCheck(result, __FILE__, __LINE__, false)  
#define NVRTC_ASSERT(result) nvrtcCheck(result, __FILE__, __LINE__, true)

#define CU_WARN(result) cuCheck(result, __FILE__, __LINE__, false)
#define CU_ASSERT(result) cuCheck(result, __FILE__, __LINE__, true)

#define CUDA_WARN(result) cudaCheck(result, __FILE__, __LINE__, false)
#define CUDA_ASSERT(result) cudaCheck(result, __FILE__, __LINE__, true);

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)