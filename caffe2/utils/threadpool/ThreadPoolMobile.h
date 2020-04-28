#pragma once

//
#ifndef USE_INTERNAL_THREADPOOL_IMPL
#include <pthreadpool.h>
#endif

#include <functional>
#include <memory>

namespace caffe2 {

class MobileThreadPool final {
public:
  explicit MobileThreadPool(size_t thread_count);
  ~MobileThreadPool() = default;

  MobileThreadPool(const MobileThreadPool&) = delete;
  MobileThreadPool& operator=(const MobileThreadPool&) = delete;

  MobileThreadPool(MobileThreadPool&&) = default;
  MobileThreadPool& operator=(MobileThreadPool&&) = default;

  size_t get_thread_count() const;
  void set_thread_count(size_t thread_count);

  // Run, in parallel, function fn(task_id) over task_id in range [0, range).
  // This function is blocking.  All input is processed by the time it returns.
  void run(const std::function<void(size_t)>& fn, size_t range);

private:
  friend pthreadpool_t mobile_pthreadpool();

private:
#ifndef USE_INTERNAL_THREADPOOL_IMPL
  324
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
#endif
};

// Return a singleton instance of MobileThreadPool for ATen/TH multithreading.
MobileThreadPool* mobile_threadpool();

// Exposes the underlying implementation of MobileThreadPool.
// Only for use in external libraries so as to unify mobile threading across
// internal (i.e. ATen, etc.) and external (e.g. NNPACK, QNNPACK, XNNPACK)
// use cases.
pthreadpool_t mobile_pthreadpool();

} // namespace caffe2
