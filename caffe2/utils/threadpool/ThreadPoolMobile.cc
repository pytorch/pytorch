#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/pthreadpool.h>

namespace caffe2 {

caffe2::ThreadPool* mobile_threadpool() {
#ifdef C10_MOBILE
  static std::unique_ptr<caffe2::ThreadPool> thread_pool =
      caffe2::ThreadPool::defaultThreadPool();
  return thread_pool.get();
#else
  return nullptr;
#endif
}

pthreadpool_t mobile_pthreadpool() {
  return reinterpret_cast<pthreadpool_t>(mobile_threadpool());
}
} // namespace caffe2
