#pragma once

#include "cudnn-wrapper.h"
#include <string>
#include <stdexcept>
#include <sstream>

struct THCState;

namespace at { namespace native {

class cudnn_exception : public std::runtime_error {
public:
  cudnnStatus_t status;
  cudnn_exception(cudnnStatus_t status, const char* msg)
      : std::runtime_error(msg)
      , status(status) {}
  cudnn_exception(cudnnStatus_t status, const std::string& msg)
      : std::runtime_error(msg)
      , status(status) {}
};

inline void CUDNN_CHECK(cudnnStatus_t status)
{
  if (status != CUDNN_STATUS_SUCCESS) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) {
        throw cudnn_exception(status, std::string(cudnnGetErrorString(status)) +
                ". This error may appear if you passed in a non-contiguous input.");
    }
    throw cudnn_exception(status, cudnnGetErrorString(status));
  }
}

inline void CUDA_CHECK(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::string msg("CUDA error: ");
    msg += cudaGetErrorString(error);
    throw std::runtime_error(msg);
  }
}

}}  // namespace at::cudnn
