#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <caffe2/utils/threadpool/thread_pool_guard.h>
#include <c10/util/Exception.h>

#include <atomic>

namespace {
// Number of threads in the thread-pool before fork
std::atomic<int> num_threads_before_fork{-1};

} // namespace

namespace caffe2 {

// Handler used by pthread_atfork that's executed before fork starts processing.
// Saves the value of the number of threads prior to fork, so that the Caffe2
// threadpool can be leaked, and restored via after_fork handler, after fork.
// It's done in order to prevent segfaults in worker processes, which otherwise
// segfault, as they try to handle inherited data-structures of parent's Caffe2
// thread-pool, although those threads don't exist in child processes.
// Ref: https://github.com/pytorch/pytorch/issues/54752#issuecomment-810315302
static void before_fork() {
#ifdef USE_PTHREADPOOL
  PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
  num_threads_before_fork.store(pool->get_thread_count());
  pool->set_thread_count(1);
#endif
 }

// Handler used by pthread_atfork that's executed after fork starts processing
// Restores the Caffe2 thread-pool after fork.
static void after_fork() {
#ifdef USE_PTHREADPOOL
  PThreadPool* const pool = caffe2::pthreadpool();
  TORCH_INTERNAL_ASSERT(pool, "Invalid thread pool!");
  pool->set_thread_count(num_threads_before_fork.load());
#endif
}

PThreadPool::PThreadPool(const size_t thread_count)
    : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {}

size_t PThreadPool::get_thread_count() const {
  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

void PThreadPool::set_thread_count(const size_t thread_count) {
  std::lock_guard<std::mutex> lock{mutex_};

  // As it stands, pthreadpool is an entirely data parallel framework with no
  // support for task parallelism.  Hence, all functions are blocking, and no
  // user-provided tasks can be in flight when the control is returned to the
  // user of the API, which means re-initializing the library, without the
  // need to wait on any pending tasks, is all one needs to do to re-adjust
  // the thread count.
  threadpool_.reset(pthreadpool_create(thread_count));
}

void PThreadPool::run(
    const std::function<void(size_t)>& fn,
    const size_t range) {
  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");

  struct Context final {
    const std::function<void(size_t)>& fn;
  } context{
      fn,
  };

  pthreadpool_parallelize_1d(
      threadpool_.get(),
      // Note: pthreadpool_parallelize_1d() is a blocking function.  The
      // function pointer to this lambda passed on to
      // pthreadpool_parallelize_1d() cannot go out of scope until
      // pthreadpool_parallelize_1d() returns.
      [](void* const context, const size_t item) {
        reinterpret_cast<Context*>(context)->fn(item);
      },
      &context,
      range,
      0u);
}

// Forward declaration
size_t getDefaultNumThreads();

PThreadPool* pthreadpool() {
  static std::unique_ptr<PThreadPool> threadpool =
      std::make_unique<PThreadPool>(getDefaultNumThreads());
#ifndef WIN32
  // Register pthread_atfork handlers to prevent segfaults in worker processes
  // after fork.
  pthread_atfork(before_fork, after_fork, nullptr);
#endif
  return threadpool.get();
}

pthreadpool_t pthreadpool_() {
  if (caffe2::_NoPThreadPoolGuard::is_enabled()) {
    return nullptr;
  }
  PThreadPool* const threadpool = pthreadpool();
  TORCH_INTERNAL_ASSERT(
      threadpool, "Failed to acquire an instance of PThreadPool!");
  return threadpool->threadpool_.get();
}

} // namespace caffe2
