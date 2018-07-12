#pragma once

#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <THC/THCStream.h>

#include "../CUDAUtils.hpp"

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

namespace c10d {

// THCStreamGuard is a RAII guard for selecting a THCStream.
//
// It sets both the current device to the stream's device and the
// current stream in the THC state.
//
class THCStreamGuard {
 public:
  explicit THCStreamGuard(THCState* state, CUDAStream& stream)
      : device_(THCStream_device(stream.getTHCStream())), state_(state) {
    CUDADevice device(device_);
    original_ = THCState_getStream(state_);
    THCStream_retain(original_);
    THCState_setStream(state_, stream.getTHCStream());
  }

  THCStreamGuard(THCStreamGuard&& other)
      : device_(other.device_), state_(nullptr), original_(nullptr) {
    std::swap(state_, other.state_);
    std::swap(original_, other.original_);
  }

  ~THCStreamGuard() {
    if (original_ != nullptr) {
      CUDADevice device(device_);
      THCState_setStream(state_, original_);
      THCStream_free(original_);
    }
  }

 private:
  const int device_;
  THCState* state_;
  THCStream* original_;
};

} // namespace c10d
