#if AT_PARALLEL_NATIVE_MOBILE
#include <ATen/Parallel.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

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

} // namespace

namespace internal {

void _run_with_pool(const std::function<void(int, size_t)>& fn, size_t range) {
  caffe2::ThreadPool* pool = caffe2::mobile_threadpool();
  if (pool) {
    pool->run(fn, range);
  } else {
    for (size_t i = 0; i < range; ++i) {
      fn(0, i);
    }
  }
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
  caffe2::mobile_threadpool();
}

void set_num_threads(int nthreads) {
  AT_ERROR("set_num_threads is not supported for mobile.");
}

int get_num_threads() {
  caffe2::ThreadPool* pool = caffe2::mobile_threadpool();
  // caffe2::ThreadPool::getNumThreads() counts the current thread.
  return !pool ? 1 /* current thread */ : pool->getNumThreads();
}

int get_thread_num() {
  return thread_num_;
}

bool in_parallel_region() {
  return in_parallel_region_;
}

void intraop_launch(std::function<void()> func) {
  // TODO: caffe2::ThreadPool doesn't support submitting tasks separately and
  // running in parallel. Should fix it when this API becomes popular.
  func();
}

std::shared_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func) {
  // TODO: caffe2::ThreadPool doesn't support submitting tasks separately and
  // running in parallel. Should fix it when this API becomes popular.
  auto future = std::make_shared<c10::ivalue::Future>();
  func();
  future->markCompleted();
  return future;
}

} // namespace at
#endif
