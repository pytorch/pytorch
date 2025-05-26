#include <ATen/Config.h>
#if AT_PARALLEL_OPENMP || AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>
#include <ATen/ThreadLocalState.h>

#include <atomic>

namespace at {

namespace {
const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of inter-op threads set by the user;
// NOT_SET -> positive value -> CONSUMED
// (CONSUMED - thread pool is initialized)
// or
// NOT_SET -> CONSUMED
std::atomic<int> num_interop_threads{NOT_SET};

// thread pool global instance is hidden,
// users should use at::launch and get/set_num_interop_threads interface
TaskThreadPoolBase& get_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ num_interop_threads.exchange(CONSUMED),
          /* create_new */ true);
  return *pool;
}

// Factory function for ThreadPoolRegistry
std::shared_ptr<TaskThreadPoolBase> create_c10_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // For now, the only accepted device id is 0
  TORCH_CHECK(device_id == 0);
  // Create new thread pool
  TORCH_CHECK(create_new);
  return std::make_shared<PTThreadPool>(pool_size);
}

} // namespace

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, create_c10_threadpool)

void set_num_interop_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");

  int no_value = NOT_SET;
  TORCH_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads after parallel work "
      "has started or set_num_interop_threads called");
}

size_t get_num_interop_threads() {
  at::internal::lazy_init_num_threads();
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    // return default value
    return TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();
  }
}

namespace internal {
void launch_no_thread_state(std::function<void()> fn) {
#if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  intraop_launch(std::move(fn));
#else
  get_pool().run(std::move(fn));
#endif
}
} // namespace internal

void launch(std::function<void()> func) {
  // NOLINTNEXTLINE(modernize-avoid-bind)
  internal::launch_no_thread_state(std::bind([](
    const std::function<void()>& f, const ThreadLocalState& thread_locals) {
      ThreadLocalStateGuard guard(thread_locals);
      f();
    },
    std::move(func),
    ThreadLocalState()
  ));
}

} // namespace at
#endif
