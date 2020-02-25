#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <caffe2/utils/threadpool/ThreadPool.h>
#include <caffe2/utils/threadpool/pthreadpool.h>

namespace caffe2 {

// Will be unified.
pthreadpool_t xnnpack_threadpool() {
  static std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
      threadpool(pthreadpool_create(getDefaultNumThreads()), pthreadpool_destroy);
  return threadpool.get();
}

} // namespace caffe2
