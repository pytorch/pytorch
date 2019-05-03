#if AT_PARALLEL_NATIVE
#include <ATen/Parallel_native.h>
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
std::atomic<int> num_intraop_threads{-1};

// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local int thread_num_ = -1;
} // namespace

namespace internal {
TaskThreadPoolBase& _get_intraop_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ check_and_get_pool_size(num_intraop_threads.exchange(-2)),
          /* create_new */ true); // create a separate thread pool for intra-op
  return *pool;
}

void _set_in_parallel_region(bool force) {
  in_parallel_region_ = force;
}

void _set_thread_num(size_t thread_num) {
  thread_num_ = thread_num;
}

void _unset_thread_num() {
  thread_num_ = -1;
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

void set_num_threads(size_t nthreads) {
  if (nthreads == 0) {
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

size_t get_num_threads() {
  int nthreads = num_intraop_threads.load();
  // add plus one because master thread is also used in
  // parallel computation
  if (nthreads > 0) {
    return nthreads + 1;
  } else if (nthreads == -1) {
    // return default value
    return check_and_get_pool_size(-1) + 1;
  } else {
    // pool is initialized, get value from the pool
    return internal::_get_intraop_pool().size() + 1;
  }
}

int get_thread_num() {
  if (thread_num_ >= 0) {
    return thread_num_;
  }

  auto thread_num = internal::_get_intraop_pool().threadNum();
  // master thread has number zero
  if (thread_num < 0) {
    return 0;
  } else {
    // add one for threads in threadpool
    return thread_num + 1;
  }
}

bool in_parallel_region() {
  return in_parallel_region_ || internal::_get_intraop_pool().inThreadPool();
}

} // namespace at
#endif
