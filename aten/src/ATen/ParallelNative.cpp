#if AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/PTThreadPool.h>

#ifndef C10_MOBILE
#include <c10/core/thread_pool.h>
#else
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#endif // C10_MOBILE

#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {
namespace {
// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local size_t thread_num_ = 0;

void _set_in_parallel_region(bool in_region) {
  in_parallel_region_ = in_region;
}

void _set_thread_num(size_t thread_num) {
  thread_num_ = thread_num;
}

void _unset_thread_num() {
  thread_num_ = 0;
}

#ifndef C10_MOBILE

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

int _num_pool_threads(int nthreads) {
  if (nthreads == NOT_SET) {
    nthreads = intraop_default_num_threads();
  } else {
    TORCH_INTERNAL_ASSERT(nthreads > 0);
  }
  // minus one because of the master thread
  return nthreads - 1;
}

TaskThreadPoolBase& _get_intraop_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ _num_pool_threads(num_intraop_threads.exchange(CONSUMED)),
          /* create_new */ true); // create a separate thread pool for intra-op
  return *pool;
}

#endif // C10_MOBILE

} // namespace

namespace internal {

void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range) {
#ifndef C10_MOBILE
  for (size_t i = 1; i < range; ++i) {
    _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
  }
  fn(0, 0);
#else
  caffe2::ThreadPool* pool = caffe2::mobile_threadpool();
  if (pool) {
    pool->run(fn, range);
  } else {
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }
  }
#endif // C10_MOBILE
}

ParallelRegionGuard::ParallelRegionGuard(int64_t task_id) {
  _set_thread_num(task_id);
  _set_in_parallel_region(true);
}

ParallelRegionGuard::~ParallelRegionGuard() {
  _set_in_parallel_region(false);
  _unset_thread_num();
}

} // namespace internal

void init_num_threads() {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif

#ifdef TH_BLAS_MKL
  mkl_set_num_threads(1);
#endif

#ifdef C10_MOBILE
  caffe2::mobile_threadpool();
#endif
}

void set_num_threads(int nthreads) {
#ifndef C10_MOBILE
  TORCH_CHECK(nthreads > 0, "Expected positive number of threads");
  int no_value = NOT_SET;
  TORCH_CHECK(num_intraop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads "
      "after parallel work has started or after set_num_threads call");
#else
  TORCH_CHECK(false, "set_num_threads is not supported for mobile.");
#endif // C10_MOBILE
}

int get_num_threads() {
#ifndef C10_MOBILE
  // not initializing pool unnecessarily,
  // because pool cannot be resized after initialization
  int nthreads = num_intraop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    return intraop_default_num_threads();
  } else {
    TORCH_INTERNAL_ASSERT(nthreads == CONSUMED);
    return _get_intraop_pool().size() + 1;
  }
#else
  caffe2::ThreadPool* pool = caffe2::mobile_threadpool();
  // caffe2::ThreadPool::getNumThreads() counts the current thread.
  return !pool ? 1 /* current thread */ : pool->getNumThreads();
#endif // C10_MOBILE
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
#ifndef C10_MOBILE
  return in_parallel_region_ || (
    num_intraop_threads.load() == CONSUMED &&
    _get_intraop_pool().inThreadPool()
  );
#else
  return in_parallel_region_;
#endif // C10_MOBILE
}

void intraop_launch(std::function<void()> func) {
#ifndef C10_MOBILE
  if (!in_parallel_region() && get_num_threads() > 1) {
    _get_intraop_pool().run(func);
  } else {
    // execute inline if we're in parallel region
    func();
  }
#else
  // TODO: caffe2::ThreadPool doesn't support submitting tasks separately and
  // running in parallel. Should fix it when this API becomes popular.
  func();
#endif // C10_MOBIL
}

std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
#ifndef C10_MOBILE
  auto future = std::make_shared<c10::ivalue::Future>(NoneType::get());
  if (!in_parallel_region() && get_num_threads() > 1) {
    _get_intraop_pool().run(
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
#else
  // TODO: caffe2::ThreadPool doesn't support submitting tasks separately and
  // running in parallel. Should fix it when this API becomes popular.
  auto future = std::make_shared<c10::ivalue::Future>(NoneType::get());
  func();
  future->markCompleted();
  return future;
#endif // C10_MOBILE
}

} // namespace at
#endif
