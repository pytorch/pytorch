#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include "cudnn-wrapper.h"

struct THCState;

namespace at {
namespace native {

class cudnn_exception : public std::runtime_error {
 public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg), status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg), status(status) {}
};

inline void cudnnCheck(cudnnStatus_t status, const char* file, int line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    std::string msg("CuDNN error: ");
    msg += cudnnGetErrorString(status);
    msg = msg + " (" + file + ":" + std::to_string(line) + ")";
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
      msg += ". This error may appear if you passed in a non-contiguous input.";
      throw cudnn_exception(status, msg);
    }
    throw cudnn_exception(status, msg);
  }
}

#define CUDNN_CHECK(ERROR) ::at::native::cudnnCheck(ERROR, __FILE__, __LINE__)

inline void cudaCheck(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    msg = msg + " (" + file + ":" + std::to_string(line) + ")";
    throw std::runtime_error(msg);
  }
}

#define CUDA_CHECK(ERROR) ::at::native::cudaCheck(ERROR, __FILE__, __LINE__)

} // namespace native
} // namespace at
