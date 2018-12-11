#include <ATen/cudnn/Handle.h>

#include <ATen/cuda/Exceptions.h>

#include <unordered_map>
#include <stack>
#include <mutex>

// TODO: Get rid of the mutex, and just initialize these
// handles in at::Context along with lazy CUDA initialization

namespace at { namespace native {

namespace {

struct Handle {
  bool responsible_for_destruction;
  cudnnHandle_t handle;
  Handle() : handle(NULL), responsible_for_destruction(true) {
    AT_CUDNN_CHECK(cudnnCreate(&handle));
  }
  // std::vector.emplace() and push_back() may route through temporaries and call
  // copy/move constructors along the way.  If this is the case, we don't want
  // the destructors of temporaries to call cudnnDestroy on the handle.
  // We can achieve safety (for the narrow case of stashing within STL containers)
  // by defining copy and move constructors that transfer cudnnDestroy
  // responsibility to the latest constructed object.  This is NOT a substitute for
  // full-blown reference counting, but reference counting may be overkill here.
  // Another alternative is to wrap the saved Handles in unique_ptrs, i.e.,
  // unordered_map<int, vector<unique_ptr<Handle>>> created_handles;
  void transfer(Handle& rhs) {
    responsible_for_destruction = true;
    handle = rhs.handle;
    rhs.responsible_for_destruction = false;
  }
  Handle(Handle& rhs) { transfer(rhs); }
  Handle(Handle&& rhs) { transfer(rhs); }
  Handle& operator=(Handle rhs) { transfer(rhs); }
  ~Handle() {
    if (handle && responsible_for_destruction) {
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
// but are never destroyed until the end of the process.
// The maximum number of handles this process will create is equal to the high-water
// mark of the number of concurrently active threads that have requested handles.
// When threads terminate, they release their handles back into the pool for reuse.
// Otherwise, new handles would be created every time new threads were spawned,
// resulting in poor performance for Python modules that repeatedly or frequently
// spawned new sets of threads (like DataParallel, which creates a new set of threads
// for each forward pass).
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
std::unordered_map<int, std::vector<Handle>> created_handles;
std::unordered_map<int, std::vector<cudnnHandle_t>> available_handles;

// PoolWindow lazily creates and caches the handles that a particular thread is using,
// so in the common case handle access doesn't incur either handle creation or a mutex lock.
class PoolWindow
{
  public:
  PoolWindow(){}
  ~PoolWindow(){ release(); }

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
      my_handles[device] = available_handles[device].back();
      available_handles[device].pop_back();
    }
    else
    {
      // In local testing, I do observe that emplace_back sometimes routes through temporaries
      // that incur copy-constructor and destructor calls.  See comments in Handle above.
      created_handles[device].emplace_back(); // no arguments to Handle constructor
      my_handles[device] = created_handles[device].back().handle;
    }

    return my_handles[device];
  }

  private:
  // Stores the per-device handles currently owned by this thread
  std::unordered_map<int, cudnnHandle_t> my_handles;

  // Called by the destructor.  Releases this thread's handles back into the pool.
  void release()
  {
    std::lock_guard<std::mutex> guard(mutex);
    for(auto d_h : my_handles)
      available_handles[d_h.first].push(d_h.second);
  }
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
