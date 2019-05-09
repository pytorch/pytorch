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
// Number of threads set by the user
// -1 -> positive value -> -2
// or
// -1 -> -2
// Meaning:
//  - -1 - pool not initialized, user value is not set
//  - positive value - pool not initialized, user value set
//  - -2 - pool is initialized
std::atomic<int> num_intraop_threads{-1};

// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local size_t thread_num_ = 0;
} // namespace

namespace internal {

TaskThreadPoolBase& _get_intraop_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          // minus one because of the master thread
          /* pool_size */ num_intraop_threads.exchange(-2) - 1,
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

//TODO: use OMP and MKL env. vars as default values
void init_num_threads() {
  #ifdef _OPENMP
  omp_set_num_threads(1);
  #endif

  #ifdef TH_BLAS_MKL
  mkl_set_num_threads(1);
  #endif
}

void set_num_threads(int nthreads) {
  if (nthreads <= 0) {
    throw std::runtime_error(
      "Expected positive number of threads");
  }
  int no_value = -1;
  if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    throw std::runtime_error(
      "Error: cannot set number of interop threads "
      "after parallel work has started or after set_num_threads call");
  }
}

int get_num_threads() {
  // not initializing pool unnecessarily,
  // because pool cannot be resized after initialization
  int nthreads = num_intraop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == -1) {
    // add plus one because master thread is also used in
    // parallel computation
    return TaskThreadPoolBase::defaultNumThreads() + 1;
  } else {
    AT_ASSERT(nthreads == -2);
    return internal::_get_intraop_pool().size() + 1;
  }
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
  return in_parallel_region_ || (
    // pool is already created or single thread case
    num_intraop_threads.load() == -2 &&
    internal::_get_intraop_pool().inThreadPool()
  );
}

} // namespace at
#endif
