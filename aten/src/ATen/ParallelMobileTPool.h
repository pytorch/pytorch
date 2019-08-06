#ifdef C10_MOBILE
#pragma once
#include "caffe2/utils/threadpool/ThreadPool.h"
#include "caffe2/utils/threadpool/pthreadpool.h"

namespace at {
namespace {
static std::unique_ptr<caffe2::ThreadPool> thread_pool_ = nullptr;
static std::mutex thread_pool_creation_mutex_;
} // namespace
inline pthreadpool_t mobile_threadpool() {
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);
  if (!thread_pool_) {
    thread_pool_ = caffe2::ThreadPool::defaultThreadPool();
  }
  return reinterpret_cast<pthreadpool_t>(thread_pool_.get());
}
} // namespace at
#endif
