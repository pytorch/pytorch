#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <caffe2/utils/threadpool/thread_pool_guard.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <c10/util/Exception.h>

namespace {
// After fork, the child process inherits the data-structures of the parent
// process' thread-pool, but since those threads don't exist, the thread-pool
// is corrupt. It's leaked in order to prevent segfaults.
// Ref: https://github.com/pytorch/pytorch/issues/54752#issuecomment-810315302
bool leak_corrupted_threadpool = false;

void child_atfork() {
  leak_corrupted_threadpool = true;
}

} // namespace

namespace caffe2 {

PThreadPool::PThreadPool(const size_t thread_count)
    : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {}

size_t PThreadPool::get_thread_count() const {
  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

void PThreadPool::set_thread_count(const size_t thread_count) {
  // No need to do anything if the count is same
  if (thread_count == get_thread_count()) {
    return;
  }

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
  // Run on same thread if _NoPThreadPoolGuard guard is enabled
  if (caffe2::_NoPThreadPoolGuard::is_enabled()) {
    for (size_t i = 0; i < range; ++i) {
      fn(i);
    }
    return;
  }

  std::lock_guard<std::mutex> lock{mutex_};

  TORCH_INTERNAL_ASSERT(!caffe2::_NoPThreadPoolGuard::is_enabled(), "Inside a threadpool guard!");
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

PThreadPool* pthreadpool(size_t thread_count) {
  static auto threadpool =
    std::make_unique<PThreadPool>(thread_count);
#if !(defined(WIN32))
  static std::once_flag flag;
  std::call_once(flag, []() {
    pthread_atfork(nullptr, nullptr, child_atfork);
  });
#endif
  if (C10_UNLIKELY(leak_corrupted_threadpool)) {
    leak_corrupted_threadpool = false;
    if (auto leaked = threadpool.release()) {
      auto num_threads = leaked->get_thread_count();
      // NOLINTNEXTLINE(modernize-make-unique)
      threadpool.reset(new PThreadPool(num_threads));
    }
  }
  return threadpool.get();
}

PThreadPool* pthreadpool() {
  return pthreadpool(getDefaultNumThreads());
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
