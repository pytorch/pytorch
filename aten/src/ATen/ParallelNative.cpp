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

// Run lambda function `fn` over `task_id` in [0, `range`) with threadpool.
// `fn` will be called with params: (thread_pool_task_id, task_id).
void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range) {
#ifndef C10_MOBILE
  for (size_t i = 1; i < range; ++i) {
    _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
  }
  // Run the first task on the current thread directly.
  fn(0, 0);
#else
  caffe2::ThreadPool* pool = caffe2::mobile_threadpool();
  if (pool) {
    // caffe2::ThreadPool can utilize the current thread.
    pool->run(fn, range);
  } else {
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }
  }
#endif // C10_MOBILE
}

// RAII guard helps to support in_parallel_region() and get_thread_num() API.
struct ParallelRegionGuard {
  ParallelRegionGuard(int64_t task_id) {
    _set_thread_num(task_id);
    _set_in_parallel_region(true);
  }

  ~ParallelRegionGuard() {
    _set_in_parallel_region(false);
    _unset_thread_num();
  }
};

} // namespace

namespace internal {

void _parallel_run(
  const int64_t begin,
  const int64_t end,
  const int64_t grain_size,
  const std::function<void(int64_t, int64_t, size_t)>& f) {
  size_t num_tasks, chunk_size;
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  std::vector<std::shared_ptr<c10::ivalue::Future>> futures(num_tasks);
  for (size_t task_id = 0; task_id < num_tasks; ++task_id) {
    futures[task_id] = std::make_shared<c10::ivalue::Future>(c10::NoneType::get());
  }
  auto task = [f, &eptr, &err_flag, &futures, begin, end, chunk_size]
      (int /* unused */, size_t task_id) {
    int64_t local_start = begin + task_id * chunk_size;
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      try {
        ParallelRegionGuard guard(task_id);
        f(local_start, local_end, task_id);
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
    futures[task_id]->markCompleted();
  };
  _run_with_pool(task, num_tasks);

  // Wait for all tasks to finish.
  for (size_t task_id = 0; task_id < num_tasks; ++task_id) {
    futures[task_id]->wait();
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
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
  if (!num_intraop_threads.compare_exchange_strong(no_value, nthreads)) {
    // num_intraop_threads either stores a positive integer or CONSUMED,
    // check that requested size is the same as the current one
    int stored_nthreads = num_intraop_threads.load();
    if (stored_nthreads <= 0) {
      // plus one because of master thread
      stored_nthreads = _get_intraop_pool().size() + 1;
    }
    if (stored_nthreads != nthreads) {
      TORCH_WARN(
        "Cannot set number of intraop threads "
        "after parallel work has started or after set_num_threads call "
        "when using native parallel backend");
    }
  }
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
  return !pool || in_parallel_region() ? 1 /* current thread */ : pool->getNumThreads();
#endif // C10_MOBILE
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
#ifndef C10_MOBILE
  return in_parallel_region_ || (
    num_intraop_threads.load() == CONSUMED &&
    // Needed as intraop_launch() doesn't set in_parallel_region().
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
#endif // C10_MOBILE
}

std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
#ifndef C10_MOBILE
  auto future = std::make_shared<c10::ivalue::Future>(c10::NoneType::get());
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
