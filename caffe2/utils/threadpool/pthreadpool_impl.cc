#include "caffe2/utils/threadpool/pthreadpool.h"
#include "caffe2/utils/threadpool/ThreadPool.h"


//
// External API
//

void pthreadpool_compute_1d(
    pthreadpool_t threadpool,
    pthreadpool_function_1d_t function,
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

size_t pthreadpool_get_threads_count_c2(pthreadpool_t threadpool) {
  return reinterpret_cast<caffe2::ThreadPool*>(threadpool)->getNumThreads();
}

pthreadpool_t pthreadpool_create_c2(size_t threads_count) {
  std::mutex thread_pool_creation_mutex_;
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);

  return reinterpret_cast<pthreadpool_t>(new caffe2::ThreadPool(threads_count));
}

void pthreadpool_destroy_c2(pthreadpool_t pthreadpool) {
  if (pthreadpool) {
    caffe2::ThreadPool* threadpool =
        reinterpret_cast<caffe2::ThreadPool*>(pthreadpool);
    delete threadpool;
  }
}
