#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include "caffe2/utils/threadpool/ThreadPool.h"
#include <vector>

#ifdef C10_MOBILE
std::unique_ptr<caffe2::ThreadPool> ThreadPoolMobile::thread_pool_ = nullptr;
std::mutex ThreadPoolMobile::thread_pool_creation_mutex_;

pthreadpool_t ThreadPoolMobile::qnnpack_threadpool() {
    std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);
    if (!thread_pool_) {
      thread_pool_ = caffe2::ThreadPool::defaultThreadPool();
    }
    return reinterpret_cast<pthreadpool_t>(thread_pool_.get());
}
#else
pthreadpool_t ThreadPoolMobile::qnnpack_threadpool() {
  return nullptr;
}
#endif
