#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/pthreadpool.h>

namespace caffe2 {

pthreadpool_t mobile_threadpool() {
#ifdef C10_MOBILE
  static std::unique_ptr<caffe2::ThreadPool> thread_pool =
      caffe2::ThreadPool::defaultThreadPool();
  return reinterpret_cast<pthreadpool_t>(thread_pool.get());
#else
  return nullptr;
#endif
}
} // namespace caffe2
