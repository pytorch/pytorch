#if AT_PARALLEL_NATIVE_TBB
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#include <atomic>
#include <mutex>

#include "tbb/tbb.h"
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace {
static thread_local tbb::task_scheduler_init tbb_init_(intraop_default_num_threads());
static thread_local tbb::task_group tg_;

std::mutex global_thread_mutex_;
std::shared_ptr<tbb::global_control> global_thread_limit_ = nullptr;
std::atomic<int> num_intraop_threads_{-1};

void _internal_set_num_threads(int nthreads) {
  TORCH_INTERNAL_ASSERT(nthreads > 0);
  {
    std::unique_lock<std::mutex> lk(global_thread_mutex_);
    global_thread_limit_ = std::make_shared<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, nthreads);
    num_intraop_threads_.store(nthreads);
  }
  if (tbb_init_.is_active()) {
    tbb_init_.terminate();
  }
  tbb_init_.initialize(nthreads);
}
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
  _internal_set_num_threads(nthreads);
}

void set_num_threads(int nthreads) {
  TORCH_CHECK(nthreads > 0);

  _internal_set_num_threads(nthreads);
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
  if (get_num_threads() > 1) {
    tg_.run(func);
  } else {
    func();
  }
}

std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
  auto future = std::make_shared<c10::ivalue::Future>(NoneType::get());
  if (get_num_threads() > 1) {
    tg_.run(
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
