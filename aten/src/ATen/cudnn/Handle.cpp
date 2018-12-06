#include "Handle.h"

#include "ATen/cuda/Exceptions.h"

#include <unordered_map>
#include <stack>
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

// Handles are lazily created as different threads request them,
// but are never destroyed until the end of the program.
// This is the desired behavior: We want unused handles to stick around,
// so they can be reused by other threads later.
//
// To prevent potential deadlocks, we explicitly choose not to cap the number
// of handles that are created per device.
// Example of danger: If we cap the max handles at 4, and 5 threads are sharing a device,
// only 4 can make forward progress at any time. The other 4 will not release their
// handles until they exit, so the fifth cannot make progress until then.  This is
// not a problem...UNLESS all 5 threads attempt some sort of synchronization at an
// intermediate point (ie, before any of them have exited).  We have no way to anticipate
// or enforce that user threads will not attempt such intermediate synchronization.
// The only way to ensure safety is to avoid imposing a cap on the number of handles.
std::unordered_map<int, std::stack<Handle>> created_handles;
std::unordered_map<int, std::stack<cudnnHandle_t>> available_handles;

struct PoolWindow
{
  std::unordered_map<int, cudnnHandle_t> my_handles;

  PoolWindow(){}

  cudnnHandle_t reserve(int device)
  {
    // If this thread already has a handle for this device, return it
    if(my_handles.find(device) != my_handles.end())
      return my_handles[device];

    // otherwise, either grab a handle from the pool if one is available,
    // or if not, create a new one.
    std::lock_guard<std::mutex> guard(mutex);

    if(available_handles[device].size() > 0)
    {
      my_handles[device] = available_handles[device].top();
      available_handles[device].pop();
    }
    else
    {
      created_handles[device].emplace(); // no arguments to Handle constructor
      my_handles[device] = created_handles[device].top().handle;
    }

    return my_handles[device];
  }

  void release()
  {
    std::lock_guard<std::mutex> guard(mutex);
    for(auto d_h : my_handles)
      available_handles[d_h.first].push(d_h.second);
  }

  ~PoolWindow(){ release(); }
};

// This will be destroyed when the thread terminates,
// releasing its reserved handles back to the pool.
thread_local PoolWindow myPoolWindow;
}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  return myPoolWindow.reserve(device);
}

}} // namespace at::cudnn
