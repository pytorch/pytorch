#pragma once

#ifdef USE_PTHREADPOOL

#ifdef USE_INTERNAL_PTHREADPOOL_IMPL
#include <caffe2/utils/threadpool/pthreadpool.h>
#else
#include <pthreadpool.h>
#endif

#include <functional>
#include <memory>
#include <mutex>

namespace caffe2 {

class PThreadPool final {
 public:
  explicit PThreadPool(size_t thread_count);
  ~PThreadPool() = default;

  PThreadPool(const PThreadPool&) = delete;
  PThreadPool& operator=(const PThreadPool&) = delete;

  PThreadPool(PThreadPool&&) = delete;
  PThreadPool& operator=(PThreadPool&&) = delete;

  size_t get_thread_count() const;
  void set_thread_count(size_t thread_count);

  // Run, in parallel, function fn(task_id) over task_id in range [0, range).
  // This function is blocking.  All input is processed by the time it returns.
  void run(const std::function<void(size_t)>& fn, size_t range);

 private:
  friend pthreadpool_t pthreadpool_();

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};

// Return a singleton instance of PThreadPool for ATen/TH multithreading.
PThreadPool* pthreadpool();
PThreadPool* pthreadpool(size_t thread_count);

// Exposes the underlying implementation of PThreadPool.
// Only for use in external libraries so as to unify threading across
// internal (i.e. ATen, etc.) and external (e.g. NNPACK, QNNPACK, XNNPACK)
// use cases.
pthreadpool_t pthreadpool_();

} // namespace caffe2

#endif /* USE_PTHREADPOOL */
