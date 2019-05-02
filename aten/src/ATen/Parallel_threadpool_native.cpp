#ifdef AT_PARALLEL_OPENMP
#include <ATen/Parallel.h>

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
          /* create_new */ false);
  return *pool;
}

std::shared_ptr<TaskThreadPoolBase> createC10ThreadPool(
    int device_id,
    int pool_size,
    bool create_new) {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      std::make_shared<PTThreadPool>(
          pool_size > 0 ? pool_size : std::thread::hardware_concurrency());
  // For now, the only accepted device id is 0
  // for the JIT inter-op pool (CPU),
  AT_ASSERT(device_id == 0);
  // we use the shared thread pool
  AT_ASSERT(!create_new);
  // and the size does not change
  AT_ASSERT(pool->size() == pool_size);
  return pool;
}

} // namespace

PTThreadPool::PTThreadPool(
    std::size_t pool_size,
    int numa_node_id)
    : c10::ThreadPool(pool_size, numa_node_id) {}

void PTThreadPool::init_thread() {
  c10::setThreadName("PTThreadPool");
  at::init_num_threads();
}

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, createC10ThreadPool);

void set_num_interop_threads(size_t nthreads) {
  if (nthreads == 0) {
    return;
  }

  int no_value = -1;
  if (!num_interop_threads.compare_exchange_strong(no_value, nthreads)) {
    throw std::runtime_error(
      "Error: cannot set number of interop threads "
      "after parallel work has started");
  }
}

size_t get_num_interop_threads() {
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == -1) {
    return std::thread::hardware_concurrency();
  } else {
    return get_pool().size();
  }
}

void launch(const std::function<void()>& func) {
  get_pool().run(func);
}

} // namespace at
#endif
