#include <ATen/ParallelMobileThreadPool.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/pthreadpool.h>

namespace at {
#ifdef C10_MOBILE
namespace {
std::unique_ptr<caffe2::ThreadPool> thread_pool_ = nullptr;
std::mutex thread_pool_creation_mutex_;
} // namespace
pthreadpool_t mobile_threadpool() {
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);
  if (!thread_pool_) {
    thread_pool_ = caffe2::ThreadPool::defaultThreadPool();
  }
  return reinterpret_cast<pthreadpool_t>(thread_pool_.get());
}
#else
pthreadpool_t mobile_threadpool() {
  return nullptr;
}
#endif
} // namespace at
