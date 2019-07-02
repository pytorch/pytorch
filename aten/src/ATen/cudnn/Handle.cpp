#include <ATen/cudnn/Handle.h>

#include <ATen/cuda/Exceptions.h>

#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>
#include <memory>

namespace at { namespace native {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle(bool create = false) : handle(nullptr)
  {
    if(create)
      AT_CUDNN_CHECK(cudnnCreate(&handle));
  }
  // std::vector.emplace() and push_back() may route through temporaries and call
  // copy/move constructors along the way.  If this is the case, we don't want
  // the destructors of temporaries to call cudnnDestroy on the handle.
  // We can achieve safety (for the narrow case of stashing within std::vectors)
  // by making Handle moveable but not copyable, and transferring handle ownership
  // to the latest constructed object.  This is not a substitute for full-blown
  // reference counting, but reference counting may be overkill here.
  // Another alternative is to wrap the saved Handles in unique_ptrs, i.e.,
  // unordered_map<int, vector<unique_ptr<Handle>>> created_handles;
  Handle(const Handle& rhs) = delete;
  // Following https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
  Handle(Handle&& rhs) : Handle() { std::swap(handle, rhs.handle); }
  // operator= takes argument by value
  Handle& operator=(Handle rhs) { std::swap(handle, rhs.handle); return *this; }
  ~Handle() {
    if(handle)
    {
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
// The maximum number of handles this process will create for each device is equal
// to the high-water mark of the number of concurrently active threads that request
// handles for that device.
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
      // that incur move-constructor and destructor calls.  See comments in Handle above.
      created_handles[device].emplace_back(true /*create*/);
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
    // The conditional check below is certainly worthwhile for efficiency.
    // However, it also serves another purpose:  Without the conditional,
    // as of cuda V9.0.176 and torch.backends.cudnn.version() = 7005,
    // we observe weird nondeterministic hangs on Windows when the process first
    // attempts to create a cudnn handle.  Example with added debug print statements:
    // https://ci.pytorch.org/jenkins/job/pytorch-builds/job/pytorch-win-ws2016-cuda9-cudnn7-py3-test2/19238/console
    // The print statements reveal that when these hangs occur, the thread that is attempting
    // to create a cudnn handle for the first time hangs on the call to cudnnCreate itself.
    // These hangs have never manifested on anything but that particular Windows build.
    // All other builds seem fine.
    if(my_handles.size() > 0)
    {
      std::lock_guard<std::mutex> guard(mutex);
      for(auto d_h : my_handles)
        available_handles[d_h.first].push_back(d_h.second);
    }
  }
};

// This will be destroyed when the thread terminates,
// releasing its reserved handles back to the pool.
thread_local std::unique_ptr<PoolWindow> myPoolWindow;
}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  if (!myPoolWindow)
    myPoolWindow.reset(new PoolWindow());

  return myPoolWindow->reserve(device);
}

}} // namespace at::cudnn
