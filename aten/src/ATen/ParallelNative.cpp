#include <ATen/Config.h>
#if AT_PARALLEL_NATIVE
#include <ATen/Parallel.h>
#include <ATen/ParallelFuture.h>
#include <ATen/PTThreadPool.h>

#ifndef C10_MOBILE
#include <c10/core/thread_pool.h>
#include <c10/util/irange.h>
#else
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif // C10_MOBILE

#include <atomic>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at {
namespace {
// used with _set_in_parallel_region to mark master thread
// as in parallel region while executing parallel primitives
thread_local bool in_parallel_region_ = false;

// thread number (task_id) set by parallel primitive
thread_local int thread_num_ = 0;

void _set_in_parallel_region(bool in_region) {
  in_parallel_region_ = in_region;
}

}  // namespace (anonymous)

namespace internal {
void set_thread_num(int thread_num) {
  thread_num_ = thread_num;
}
}

namespace {
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
  for (const auto i : c10::irange(1, range)) {
    _get_intraop_pool().run([fn, i]() { fn((int)i, i); });
  }
  // Run the first task on the current thread directly.
  fn(0, 0);
#else
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");

  pool->run(
    // PThreadPool::run() is blocking.  A std::function [const] reference to
    // this lambda cannot go out of scope before PThreadPool::run() returns.
    [&fn](const size_t task_id) {
      fn(0 /* unused */, task_id);
    }, range);
#endif // C10_MOBILE
}

// RAII guard helps to support in_parallel_region() and get_thread_num() API.
struct ParallelRegionGuard {
  ParallelRegionGuard(int task_id) {
    internal::set_thread_num(task_id);
    _set_in_parallel_region(true);
  }

  ~ParallelRegionGuard() {
    _set_in_parallel_region(false);
    _unset_thread_num();
  }
};

} // namespace

namespace internal {

inline std::tuple<size_t, size_t> calc_num_tasks_and_chunk_size(
    int64_t begin, int64_t end, int64_t grain_size) {
  if ((end - begin) < grain_size) {
    return std::make_tuple(1, std::max((int64_t)0, end - begin));
  }
  // Choose number of tasks based on grain size and number of threads.
  size_t chunk_size = divup((end - begin), get_num_threads());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max((size_t)grain_size, chunk_size);
  size_t num_tasks = divup((end - begin), chunk_size);
  return std::make_tuple(num_tasks, chunk_size);
}

void invoke_parallel(
  const int64_t begin,
  const int64_t end,
  const int64_t grain_size,
  const std::function<void(int64_t, int64_t)>& f) {
  at::internal::lazy_init_num_threads();

  size_t num_tasks = 0, chunk_size = 0;
  std::tie(num_tasks, chunk_size) =
      internal::calc_num_tasks_and_chunk_size(begin, end, grain_size);

  struct {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    std::mutex mutex;
    volatile size_t remaining{0};
    std::condition_variable cv;
  } state;

  auto task = [f, &state, begin, end, chunk_size]
      (int /* unused */, size_t task_id) {
    int64_t local_start = begin + task_id * chunk_size;
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      try {
        ParallelRegionGuard guard(task_id);
        f(local_start, local_end);
      } catch (...) {
        if (!state.err_flag.test_and_set()) {
          state.eptr = std::current_exception();
        }
      }
    }
    {
      std::unique_lock<std::mutex> lk(state.mutex);
      if (--state.remaining == 0) {
        state.cv.notify_one();
      }
    }
  };
  state.remaining = num_tasks;
  _run_with_pool(std::move(task), num_tasks);

  // Wait for all tasks to finish.
  {
    std::unique_lock<std::mutex> lk(state.mutex);
    if (state.remaining != 0) {
      state.cv.wait(lk);
    }
  }
  if (state.eptr) {
    std::rethrow_exception(state.eptr);
  }
}

} // namespace internal

void init_num_threads() {
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif

#if AT_MKL_ENABLED()
  mkl_set_num_threads(1);
#endif

#ifdef C10_MOBILE
  caffe2::pthreadpool();
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
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
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
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
  pool->set_thread_count(nthreads);
#endif // C10_MOBILE
}

int get_num_threads() {
  at::internal::lazy_init_num_threads();
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
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return _get_intraop_pool().size() + 1;
  }
#else
  caffe2::PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!")
  return in_parallel_region() ? 1 /* current thread */ : pool->get_thread_count();
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
    _get_intraop_pool().run(std::move(func));
  } else {
    // execute inline if we're in parallel region
    func();
  }
#else
  // TODO: caffe2::PThreadPool only provides a data-parallel API.
  // Task parallelism is not currently supported.
  func();
#endif // C10_MOBILE
}

c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
#ifndef C10_MOBILE
  auto future = c10::make_intrusive<c10::ivalue::Future>(c10::NoneType::get());
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
  // TODO: caffe2::PThreadPool only provides a data-parallel API.
  // Task parallelism is not currently supported.
  auto future = c10::make_intrusive<c10::ivalue::Future>(c10::dynT<NoneType>());
  func();
  future->markCompleted();
  return future;
#endif // C10_MOBILE
}

} // namespace at
#endif
