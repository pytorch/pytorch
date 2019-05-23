#if AT_PARALLEL_NATIVE_TBB
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#include <atomic>

#include "tbb/tbb.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace internal {
  static thread_local tbb::task_scheduler_init tbb_init(
      tbb::task_scheduler_init::deferred);

  std::atomic<int> num_intraop_threads{-1};
}

//TODO: use OMP and MKL env. vars as default values
void init_num_threads() {
  // omp- and mkl_set_num_threads don't affect TBB versions of MKL/MKL-DNN
  #ifdef _OPENMP
  omp_set_num_threads(1);
  #endif

  #ifdef TH_BLAS_MKL
  mkl_set_num_threads(1);
  #endif

  int nthreads = internal::num_intraop_threads.load();
  if (nthreads > 0) {
    if (!internal::tbb_init.is_active()) {
      internal::tbb_init.initialize(nthreads);
    } else {
      TORCH_CHECK(tbb::this_task_arena::max_concurrency() == nthreads);
    }
  }
}

void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0);
  int no_value = -1;
  if (internal::num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    if (!internal::tbb_init.is_active()) {
      internal::tbb_init.initialize(nthreads);
      return;
    }
  }
  TORCH_CHECK(false,
    "Error: cannot set number of interop threads "
    "after parallel work has started or after set_num_threads call");
}

int get_num_threads() {
  return tbb::this_task_arena::max_concurrency();
}

int get_thread_num() {
  return tbb::this_task_arena::current_thread_index();
}

bool in_parallel_region() {
  return tbb::this_task_arena::current_thread_index() != -1;
}

} // namespace at
#endif
