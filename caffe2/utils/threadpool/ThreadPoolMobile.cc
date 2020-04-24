#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <c10/util/Exception.h>

namespace caffe2 {

size_t getDefaultNumThreads();

MobileThreadPool::MobileThreadPool(const size_t thread_count)
  : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {
}

size_t MobileThreadPool::get_thread_count() const {
  TORCH_INTERNAL_ASSERT(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

void MobileThreadPool::set_thread_count(const size_t thread_count) {
  // As it stands, pthreadpool is an entirely data parallel framework with no
  // support for task parallelism.  Hence, all functions are blocking, and no
  // user-provided tasks can be in flight when the control is returned to the
  // user of the API, which means re-initializing the library, without the
  // need to wait on any pending tasks, is all one needs to do to re-adjust
  // the thread count.
  threadpool_.reset(pthreadpool_create(thread_count));
}

void MobileThreadPool::run(
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
      // pointer to this lambda passed on to pthreadpool_parallelize_1d()cannot go
      // out of scope until pthreadpool_parallelize_1d() returns.
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

MobileThreadPool* mobile_threadpool() {
  static std::unique_ptr<MobileThreadPool> threadpool =
      std::make_unique< MobileThreadPool>(getDefaultNumThreads());
  return threadpool.get();
}

pthreadpool_t mobile_pthreadpool() {
  MobileThreadPool* const threadpool = mobile_threadpool();
  TORCH_INTERNAL_ASSERT(threadpool, "Failed to create mobile threadpool!");
  return threadpool->threadpool_.get();
}

} // namespace caffe2
