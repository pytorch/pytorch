#include <ATen/native/mobile/internal/ThreadPool.h>
#include <c10/util/Exception.h>

namespace at {
namespace native {
namespace mobile {
namespace internal {

ThreadPool::ThreadPool(const size_t thread_count)
    : pthreadpool_(pthreadpool_create(thread_count)) {
  if (!pthreadpool_) {
    throw std::runtime_error("pthreadpool_create failed!");
  }
}

ThreadPool::~ThreadPool() {
  pthreadpool_destroy(pthreadpool_);
}

size_t ThreadPool::get_thread_count() const {
  TORCH_INTERNAL_ASSERT(pthreadpool_);
  return pthreadpool_get_threads_count(pthreadpool_);
}

void ThreadPool::run(
    const std::function<void(int, size_t)>& fn,
    const size_t range) {
  TORCH_INTERNAL_ASSERT(pthreadpool_);

  struct Context final {
    const std::function<void(int, size_t)>& fn;
  } context{
      fn,
  };

// TODO (Ashkan): Disabled until integration complete.
#if 0
  pthreadpool_parallelize_1d(
      pthreadpool_,
      [](void* const context, const size_t item) {
        const union {
          void* const as_void_ptr;
          const Context* const as_context_ptr;
        } argument{
            context,
        };

        argument.as_context_ptr->fn(0, item);
      },
      &context,
      range,
      0u);
#endif
}

pthreadpool_t ThreadPool::handle() const {
  TORCH_INTERNAL_ASSERT(pthreadpool_);
  return pthreadpool_;
}

// TODO (Ashkan)

ThreadPool& threadpool() {
  static ThreadPool threadpool_(4u);
  return threadpool_;
}

} // namespace internal
} // namespace mobile
} // namespace native
} // namespace at
