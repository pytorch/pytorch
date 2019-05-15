#if AT_PARALLEL_NATIVE_TBB
#include <ATen/Parallel.h>

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
  tbb::task_scheduler_init tbb_init(
      tbb::task_scheduler_init::deferred);
  tbb::task_arena arena;

  std::atomic<int> num_intraop_threads{-1};

  tbb::task_arena& _get_arena() {
    return arena;
  }
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
}

void set_num_threads(int nthreads) {
  if (nthreads <= 0) {
    throw std::runtime_error(
      "Expected positive number of threads");
  }
  int no_value = -1;
  if (internal::num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    if (!internal::tbb_init.is_active()) {
      internal::tbb_init.initialize(nthreads);
      internal::arena.terminate();
      internal::arena.initialize(nthreads);
      return;
    }
  }
  throw std::runtime_error(
    "Error: cannot set number of interop threads "
    "after parallel work has started or after set_num_threads call");
}

int get_num_threads() {
  return internal::arena.max_concurrency();
}

int get_thread_num() {
  auto index = internal::arena.current_thread_index();
  return (index >= 0) ? index : 0;
}

bool in_parallel_region() {
  return internal::arena.current_thread_index() != -1;
}

} // namespace at
#endif
