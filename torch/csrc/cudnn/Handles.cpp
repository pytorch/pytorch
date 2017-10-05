#include <Python.h>
#include "Handles.h"

#include <unordered_map>
#include <mutex>
#include "Exceptions.h"

namespace torch { namespace cudnn {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle() : handle(NULL) {
    CHECK(cudnnCreate(&handle));
  }
  ~Handle() {
    if (handle) {
      cudnnDestroy(handle);
    }
  }
};

std::mutex mutex;
std::unordered_map<int, Handle> handles;

}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  std::lock_guard<std::mutex> guard(mutex);
  return handles[device].handle;
}


void useCurrentStream(cudnnHandle_t handle, THCState *state)
{
  cudnnStatus_t status = cudnnSetStream(
      handle,
      THCState_getCurrentStream(state));
  if (status == CUDNN_STATUS_BAD_PARAM) {
    throw std::runtime_error(
        "cudnnSetStream returned CUDNN_STATUS_BAD_PARAM: invalid handle");
  } else if (status == CUDNN_STATUS_MAPPING_ERROR) {
    throw std::runtime_error(
        "cudnnSetStream returned CUDNN_STATUS_MAPPING_ERROR: "
        "mismatch between user stream and the cuDNN handle context");
  } else if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(
        "Could not set cuDNN to use current stream. "
        "cudnnSetStream returned " + status);
  }
}

}} // namespace torch::cudnn
