#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <c10/util/Exception.h>

namespace caffe2 {

PThreadPool::PThreadPool(const size_t thread_count)
  : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {
}

size_t PThreadPool::get_thread_count() const {
  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

void PThreadPool::set_thread_count(const size_t thread_count) {
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
  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");

  // Note: Both run() and pthreadpool_parallelize_1d() are blocking functions.
  // By definition, a reference to fn as the argument to a blocking function,
  // cannot go out of scope in either case before these functions return.

  struct Context final {
    const std::function<void(size_t)>& fn;
  } context{
      fn,
  };

  pthreadpool_parallelize_1d(
      threadpool_.get(),
      // Note: pthreadpool_parallelize_1d() is a blocking function.  The function
      // pointer to this lambda passed on to pthreadpool_parallelize_1d() cannot
      // go out of scope until pthreadpool_parallelize_1d() returns.
      [](void* const context, const size_t item) {
        const union {
          void* const as_void_ptr;
          const Context* const as_context_ptr;
        } argument{
          context,
        };

        argument.as_context_ptr->fn(item);
      },
      &context,
      range,
      0u);
}

// Forward declaration
size_t getDefaultNumThreads();

PThreadPool* pthreadpool() {
  static std::unique_ptr<PThreadPool> threadpool =
      std::make_unique< PThreadPool>(getDefaultNumThreads());
  return threadpool.get();
}

pthreadpool_t pthreadpool_() {
  PThreadPool* const threadpool = pthreadpool();
  TORCH_INTERNAL_ASSERT(threadpool, "Failed to acquire an instance of PThreadPool!");
  return threadpool->threadpool_.get();
}

} // namespace caffe2
