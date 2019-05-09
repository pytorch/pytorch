#if AT_PARALLEL_OPENMP || AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#include <atomic>
#include <thread>

namespace at {

namespace {

// Number of inter-op threads set by the user;
// Atomic transitions:
// -1 -> (atomic) -> positive value -> (atomic) -> -2
// (-2 - thread pool is initialized)
// or
// -1 -> (atomic) -> -2
std::atomic<int> num_interop_threads{-1};

// thread pool global instance is hidden,
// users should use at::launch ang get/set_num_interop_threads interface
TaskThreadPoolBase& get_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ num_interop_threads.exchange(-2),
          /* create_new */ true);
  return *pool;
}

// Factory function for ThreadPoolRegistry
std::shared_ptr<TaskThreadPoolBase> create_c10_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // For now, the only accepted device id is 0
  // for the JIT inter-op pool (CPU),
  AT_ASSERT(device_id == 0);
  // Create new thread pool
  AT_ASSERT(create_new);
  return std::make_shared<PTThreadPool>(pool_size);
}

} // namespace

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, create_c10_threadpool);

void set_num_interop_threads(int nthreads) {
  if (nthreads <= 0) {
    throw std::runtime_error("Expected positive number of threads");
  }

  int no_value = -1;
  if (!num_interop_threads.compare_exchange_strong(no_value, nthreads)) {
    throw std::runtime_error(
      "Error: cannot set number of interop threads after parallel work "
      "has started or set_num_interop_threads called");
  }
}

int get_num_interop_threads() {
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == -1) {
    // return default value
    return TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();
  }
}

void launch(const std::function<void()>& func) {
  get_pool().run(func);
}

} // namespace at
#endif
