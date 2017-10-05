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
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
}

}} // namespace torch::cudnn
