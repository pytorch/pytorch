#include "Handles.h"

#include "Exceptions.h"

#include <unordered_map>
#include <mutex>

// TODO: Get rid of the mutex, and just initialize these
// handles in at::Context along with lazy CUDA initialization

namespace at { namespace native {

namespace {

struct Handle {
  miopenHandle_t handle;
  Handle() : handle(NULL) {
    MIOPEN_CHECK(miopenCreate(&handle));
  }
  ~Handle() {
    if (handle) {
      miopenDestroy(handle);
    }
  }
};

std::mutex mutex;
std::unordered_map<int, Handle> handles;

}  // namespace


miopenHandle_t getMiopenHandle()
{
  int device;
  CUDA_CHECK(hipGetDevice(&device));

  std::lock_guard<std::mutex> guard(mutex);
  return handles[device].handle;
}

}} // namespace at::miopen
