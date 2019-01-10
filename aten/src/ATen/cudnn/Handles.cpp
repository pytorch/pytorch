#include "Handles.h"

#include "Exceptions.h"

#include <unordered_map>
#include <mutex>

// TODO: Get rid of the mutex, and just initialize these
// handles in at::Context along with lazy CUDA initialization

namespace at { namespace native {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle() : handle(NULL) {
    CUDNN_CHECK(cudnnCreate(&handle));
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

}} // namespace at::cudnn
