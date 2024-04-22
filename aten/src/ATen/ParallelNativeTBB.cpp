#include <ATen/Config.h>
#if AT_PARALLEL_NATIVE_TBB
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>
#include <ATen/PTThreadPool.h>

#include <atomic>
#include <mutex>

#include <tbb/tbb.h>
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/global_control.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at {

namespace {
static thread_local tbb::task_group tg_;
thread_local int this_thread_id{0};

std::mutex global_thread_mutex_;
std::shared_ptr<tbb::global_control> global_thread_limit_ = nullptr;
std::atomic<int> num_intraop_threads_{-1};

void _internal_set_num_threads(int nthreads) {
  TORCH_INTERNAL_ASSERT(nthreads > 0);
  {
    std::unique_lock<std::mutex> lk(global_thread_mutex_);
    // This is an antipattern and we shouldn't be constraining the number of
    // threads in library code.
    // TODO: Think of a smarter way to leverage tbb::thread_arena to limit the
    // number of slots instead of the number of threads.
    global_thread_limit_ = std::make_shared<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, nthreads);
    num_intraop_threads_.store(nthreads);
  }
}
}

void init_num_threads() {
  #ifdef _OPENMP
  omp_set_num_threads(1);
  #endif

  #if AT_MKL_ENABLED()
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
  at::internal::lazy_init_num_threads();
  return tbb::global_control::active_value(
      tbb::global_control::max_allowed_parallelism);
}

int get_thread_num() {
  return this_thread_id;
}

namespace internal {
void set_thread_num(int id) {
  this_thread_id = id;
}
}

bool in_parallel_region() {
  return tbb::this_task_arena::current_thread_index() >= 0;
}

void intraop_launch(std::function<void()> func) {
  if (get_num_threads() > 1) {
    tg_.run(func);
  } else {
    func();
  }
}

c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(NoneType::get());
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
