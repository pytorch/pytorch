#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <c10/macros/Export.h>

#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace at::cuda {
namespace {

void createCusolverDnHandle(cusolverDnHandle_t *handle) {
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(handle));
}

void destroyCusolverDnHandle(cusolverDnHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
  (void)handle; // Suppress unused variable warning
#else
    cusolverDnDestroy(handle);
#endif
}

using CuSolverDnPoolType = DeviceThreadHandlePool<cusolverDnHandle_t, createCusolverDnHandle, destroyCusolverDnHandle>;

// Per-(device, stream) cuSOLVER handles for the looped multi-stream path.
// A handle is kept on its original stream to avoid rebinding it while work may
// still be in flight. Handles are thread-local while owned, and returned on
// thread exit to a process-lifetime free list keyed by the same (device, stream)
// instead of being destroyed during teardown.
using AuxCusolverHandleKey = std::pair<c10::DeviceIndex, cudaStream_t>;

struct AuxCusolverHandlePool {
  std::mutex mutex;
  std::map<AuxCusolverHandleKey, std::vector<cusolverDnHandle_t>> free_handles;
};

AuxCusolverHandlePool& auxCusolverHandlePool() {
  static auto* pool = new AuxCusolverHandlePool();
  return *pool;
}

// Thread-local cache keyed by the actual CUDA stream. Reusing a handle on the
// same stream is ordered by stream FIFO; reusing it on a different stream could
// conflict with in-flight handle-local state.
class AuxCusolverHandleWindow {
 public:
  cusolverDnHandle_t reserve(c10::DeviceIndex device, cudaStream_t stream) {
    AuxCusolverHandleKey key = std::make_pair(device, stream);
    auto it = my_handles_.find(key);
    if (it != my_handles_.end()) {
      return it->second;
    }

    cusolverDnHandle_t handle = nullptr;
    auto& pool = auxCusolverHandlePool();
    {
      std::lock_guard<std::mutex> lock(pool.mutex);
      auto& free_handles = pool.free_handles[key];
      if (!free_handles.empty()) {
        handle = free_handles.back();
        free_handles.pop_back();
      }
    }
    if (handle == nullptr) {
      createCusolverDnHandle(&handle);
    }
    my_handles_[key] = handle;
    return handle;
  }

  ~AuxCusolverHandleWindow() {
    if (my_handles_.empty()) {
      return;
    }
    auto& pool = auxCusolverHandlePool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (const auto& [key, handle] : my_handles_) {
      pool.free_handles[key].push_back(handle);
    }
  }

 private:
  std::map<AuxCusolverHandleKey, cusolverDnHandle_t> my_handles_;
};

} // namespace

cusolverDnHandle_t getCurrentCUDASolverDnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<CuSolverDnPoolType>();
  thread_local std::unique_ptr<CuSolverDnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return handle;
}

// Return a cuSOLVER handle dedicated to a specific stream, keyed by
// (device, stream). Used by the looped multi-stream path, which launches
// independent factorizations on several auxiliary streams concurrently and
// therefore cannot share one handle across them. These per-stream handles do
// not clone state from torch.cuda.current_solver_handle(); PyTorch only sets the
// stream.
cusolverDnHandle_t getCUDASolverDnHandleForStream(cudaStream_t stream) {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  thread_local AuxCusolverHandleWindow window;
  auto handle = window.reserve(device, stream);
  TORCH_CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));
  return handle;
}

extern "C" C10_EXPORT void* at_cuda_getCurrentCUDASolverDnHandle() {
  return getCurrentCUDASolverDnHandle();
}

} // namespace at::cuda
