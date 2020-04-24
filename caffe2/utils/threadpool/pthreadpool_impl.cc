#include "caffe2/utils/threadpool/pthreadpool.h"
#include "caffe2/utils/threadpool/ThreadPool.h"


//
// External API
//

void c2_pthreadpool_compute_1d(
    c2_pthreadpool_t threadpool,
    c2_pthreadpool_function_1d_t function,
    void* argument,
    size_t range) {
  if (threadpool == nullptr) {
    /* No thread pool provided: execute function sequentially on the calling
     * thread */
    for (size_t i = 0; i < range; i++) {
      function(argument, i);
    }
    return;
  }
  reinterpret_cast<caffe2::ThreadPool*>(threadpool)
      ->run(
          [function, argument](int threadId, size_t workId) {
            function(argument, workId);
          },
          range);
}

size_t c2_pthreadpool_get_threads_count(c2_pthreadpool_t threadpool) {
  // The current fix only useful when XNNPACK calls pthreadpool_get_threads_count with nullptr.
  if (threadpool == nullptr) {
    return 1;
  }
  return reinterpret_cast<caffe2::ThreadPool*>(threadpool)->getNumThreads();
}

c2_pthreadpool_t c2_pthreadpool_create(size_t threads_count) {
  std::mutex thread_pool_creation_mutex_;
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);

  return reinterpret_cast<c2_pthreadpool_t>(new caffe2::ThreadPool(threads_count));
}

void c2_pthreadpool_destroy(c2_pthreadpool_t pthreadpool) {
  if (pthreadpool) {
    caffe2::ThreadPool* threadpool =
        reinterpret_cast<caffe2::ThreadPool*>(pthreadpool);
    delete threadpool;
  }
}
