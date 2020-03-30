#include <caffe2/utils/threadpool/pthreadpool.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <caffe2/utils/threadpool/ThreadPoolXNNPACK.h>
#include <memory>

namespace caffe2 {

// Will be unified.
pthreadpool_t xnnpack_threadpool() {
// Depending on internal implemenation vs. OSS we will link against pthreadpool_create_xnnpack
// or pthreadpool_create. This is only temporary. It will be unified soon.
#ifdef USE_INTERNAL_THREADPOOL_IMPL
  static std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy_xnnpack)>
      threadpool(pthreadpool_create_xnnpack(getDefaultNumThreads()), pthreadpool_destroy_xnnpack);
#else
  static std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
      threadpool(pthreadpool_create(getDefaultNumThreads()), pthreadpool_destroy);
#endif
  return threadpool.get();
}

} // namespace caffe2
