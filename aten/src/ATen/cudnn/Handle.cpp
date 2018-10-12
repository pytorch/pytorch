#include "Handle.h"

#include "ATen/cuda/Exceptions.h"

#include <unordered_map>
#include <mutex>

// TODO: Get rid of the mutex, and just initialize these
// handles in at::Context along with lazy CUDA initialization

namespace at { namespace native {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle() : handle(NULL) {
    AT_CUDNN_CHECK(cudnnCreate(&handle));
  }
  ~Handle() {
    if (handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and I decided to not destroy
// the handle as a workaround.
//   - @soumith
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
      cudnnDestroy(handle);
#endif
    }
  }
};

std::mutex mutex;
std::unordered_map<int, Handle> handles;

}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  std::lock_guard<std::mutex> guard(mutex);
  return handles[device].handle;
}

}} // namespace at::cudnn
