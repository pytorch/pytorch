#include <ATen/cudnn/Handle.h>

#include <ATen/cuda/Exceptions.h>

#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>
#include <iostream>

namespace at { namespace native {

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle(bool create = false) : handle(nullptr)
  {
    std::cout << "thread " << std::this_thread::get_id()
              << ", Handle(" << create << ")" << std::endl;
    if(create)
    {
      AT_CUDNN_CHECK(cudnnCreate(&handle));
      std::cout << "thread " << std::this_thread::get_id()
                << " created cudnn handle " << handle << std::endl;
    }
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
  Handle(Handle&& rhs) : Handle()
  {
    std::cout << "thread " << std::this_thread::get_id()
              << ", Handle(Handle&&)"
              << ", " << handle << ", " << rhs.handle << std::endl;
    std::swap(handle, rhs.handle);
  }
  // operator= takes argument by value
  Handle& operator=(Handle rhs)
  {
    std::cout << "thread " << std::this_thread::get_id()
              << ", operator="
              << ", " << handle << ", " << rhs.handle << std::endl;
    std::swap(handle, rhs.handle); return *this;
  }
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
      // These should only ever be called at the end of the process.
      std::cout << "thread " << std::this_thread::get_id()
                << "~Handle" << handle << std::endl;
      cudnnDestroy(handle);
      std::cout << "thread " << std::this_thread::get_id()
                << "destroyed handle" << handle << std::endl;
#endif
    }
  }
};

std::mutex mutex;
std::unordered_map<int, std::vector<Handle>> handles;

}  // namespace


cudnnHandle_t getCudnnHandle()
{
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

  std::cout << "thread " << std::this_thread::get_id()
            << ", device " << device
            << ", getCudnnHandle before mutex" << std::endl;

  std::lock_guard<std::mutex> guard(mutex);

  std::cout << "thread " << std::this_thread::get_id()
            << ", device " << device
            << ", getCudnnHandle after mutex" << std::endl;

  if(handles[device].size() > 0)
  {
    std::cout << "thread " << std::this_thread::get_id()
              << ", getCudnnHandle first case" << std::endl;
    return handles[device].back().handle;
  }
  else
  {
    std::cout << "thread " << std::this_thread::get_id()
              << ", getCudnnHandle second case" << std::endl;
    handles[device].emplace_back(true /*create*/);
    return handles[device].back().handle;
  }
}

}} // namespace at::cudnn
