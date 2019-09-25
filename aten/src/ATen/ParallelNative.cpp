#if AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {
namespace {
const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of threads set by the user
// NOT_SET -> positive value -> CONSUMED
// or
// NOT_SET -> CONSUMED
// Meaning:
//  - NOT_SET - pool not initialized, user value is not set
//  - positive value - pool not initialized, user value set
//  - CONSUMED - pool is initialized
std::atomic<int> num_intraop_threads{NOT_SET};

// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local size_t thread_num_ = 0;

int _num_pool_threads(int nthreads) {
  if (nthreads == NOT_SET) {
    nthreads = intraop_default_num_threads();
  } else {
    TORCH_INTERNAL_ASSERT(nthreads > 0);
  }
  // minus one because of the master thread
  return nthreads - 1;
}
} // namespace

namespace internal {

TaskThreadPoolBase& _get_intraop_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
          /* create_new */ true); // create a separate thread pool for intra-op
  return *pool;
}

void _set_in_parallel_region(bool in_region) {
  in_parallel_region_ = in_region;
}

void _set_thread_num(size_t thread_num) {
  thread_num_ = thread_num;
}

void _unset_thread_num() {
  thread_num_ = 0;
}

} // namespace internal

void init_num_threads() {
  #ifdef _OPENMP
  omp_set_num_threads(1);
  #endif

  #ifdef TH_BLAS_MKL
  mkl_set_num_threads(1);
  #endif
}

void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  int no_value = NOT_SET;
  TORCH_CHECK(num_intraop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads "
      "after parallel work has started or after set_num_threads call");
}

int get_num_threads() {
  // not initializing pool unnecessarily,
  // because pool cannot be resized after initialization
  int nthreads = num_intraop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    return intraop_default_num_threads();
  } else {
    TORCH_INTERNAL_ASSERT(nthreads == CONSUMED);
    return internal::_get_intraop_pool().size() + 1;
  }
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
  return in_parallel_region_ || (
    num_intraop_threads.load() == CONSUMED &&
    internal::_get_intraop_pool().inThreadPool()
  );
}

void intraop_launch(std::function<void()> func) {
  if (!in_parallel_region() && get_num_threads() > 1) {
    internal::_get_intraop_pool().run(func);
  } else {
    // execute inline if we're in parallel region
    func();
  }
}

std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
  auto future = std::make_shared<c10::ivalue::Future>(NoneType::get());
  if (!in_parallel_region() && get_num_threads() > 1) {
    internal::_get_intraop_pool().run(
      [func, future]() {
        func();
        future->markCompleted();
      }
    );
  } else {
    func();
    future->markCompleted();
  }
  return future;
}

} // namespace at
#endif
