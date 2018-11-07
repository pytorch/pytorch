#pragma once

#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

// TODO: Use AT_CHECK or similar here
#define C10D_CUDA_CHECK(condition)        \
  do {                                    \
    cudaError_t error = (condition);      \
    if (error != cudaSuccess) {           \
      std::stringstream ss;               \
      ss << "Error at: ";                 \
      ss << __FILE__;                     \
      ss << ":";                          \
      ss << __LINE__;                     \
      ss << ": ";                         \
      ss << cudaGetErrorString(error);    \
      throw std::runtime_error(ss.str()); \
    }                                     \
  } while (0)
