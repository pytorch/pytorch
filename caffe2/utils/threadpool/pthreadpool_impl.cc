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

size_t pthreadpool_get_threads_count(pthreadpool_t threadpool) {
  // The current fix only useful when XNNPACK calls pthreadpool_get_threads_count with nullptr.
  if (threadpool == nullptr) {
    return 1;
  }
  return reinterpret_cast<caffe2::ThreadPool*>(threadpool)->getNumThreads();
  // TODO: Future fix: If we keep maintaining two different threadpools.
  // Old C2 and new one for XNNPACK, then the we have two different pthreadpool pointer
  // types. One is caffe2::Thredpool*, the other is pthreadpool* (pthreadpool_new_if_impl.c)
  // XNNPACK calls pthreadpool_get_threads_count during op setup using pthreadpool*, and
  // uses _parallelize_ interface for for actual work.
  // While NNPACK uses caffe2::Threadpool*.
  // Thus if pthreadpool_get_threads_count is getting called from XNNPACK we cannot
  // reinterpret_cast it to ThreadPool. It will seg fault or worse will have unedfined behavior.
}

pthreadpool_t pthreadpool_create(size_t threads_count) {
  std::mutex thread_pool_creation_mutex_;
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);

  return reinterpret_cast<pthreadpool_t>(new caffe2::ThreadPool(threads_count));
}

void pthreadpool_destroy(pthreadpool_t pthreadpool) {
  if (pthreadpool) {
    caffe2::ThreadPool* threadpool =
        reinterpret_cast<caffe2::ThreadPool*>(pthreadpool);
    delete threadpool;
  }
}
