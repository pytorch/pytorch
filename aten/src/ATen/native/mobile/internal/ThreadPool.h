#pragma once

#include <c10/macros/Macros.h>
#include <pthreadpool.h>
#include <functional>

namespace at {
namespace native {
namespace mobile {
namespace internal {

// Do NOT use directly.  Use at::parallel_for instead.

class ThreadPool final {
 public:
  explicit ThreadPool(size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  ThreadPool(ThreadPool&&) = default;
  ThreadPool& operator=(ThreadPool&&) = default;

  size_t get_thread_count() const;
  void run(const std::function<void(int, size_t)>& fn, size_t range);

  // Internal
  pthreadpool_t handle() const;

 private:
  pthreadpool_t pthreadpool_;
};

ThreadPool& threadpool();

} // namespace internal
} // namespace mobile
} // namespace native
} // namespace at
