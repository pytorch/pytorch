#if AT_PARALLEL_NATIVE_TBB
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#include <atomic>
#include <mutex>

#include "tbb/tbb.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace {
  static thread_local tbb::task_scheduler_init tbb_init_(intraop_default_num_threads());
  std::atomic<int> num_intraop_threads_{-1};
  static thread_local tbb::task_group tg_;
}

void init_num_threads() {
  #ifdef _OPENMP
  omp_set_num_threads(1);
  #endif

  #ifdef TH_BLAS_MKL
  mkl_set_num_threads(1);
  #endif

  int nthreads = num_intraop_threads_.load();
  if (nthreads < 0) {
    nthreads = intraop_default_num_threads();
  }
  TORCH_INTERNAL_ASSERT(nthreads > 0);
  if (tbb_init_.is_active()) {
    tbb_init_.terminate();
  }
  tbb_init_.initialize(nthreads);
}

void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0);
  int no_value = -1;
  if (num_intraop_threads_.compare_exchange_strong(no_value, nthreads)) {
    if (tbb_init_.is_active()) {
      tbb_init_.terminate();
    }
    tbb_init_.initialize(nthreads);
    return;
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

void intraop_launch(std::function<void()> func) {
  tg_.run(func);
}

} // namespace at
#endif
