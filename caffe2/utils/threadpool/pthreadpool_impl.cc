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
  if (threadpool) {
    // This will have to change somehow if we keep maintaining two different threadpools.
    // Old C2 and new one for XNNPACK.
    // Issue is new XNNPACK uses old interface of pthreadpool_get_threads_count during op setup,
    // while using new _parallelize_ interface for for actual work.
    // Thus if pthreadpool_get_threads_count is getting called from XNNPACK we cannot
    // reinterpret_cast it to ThreadPool. It will seg fault or worse will have unedfined behavior.
    // Good new is that pthreadpool_get_threads_count is used only by (besides in bench/test)
    // XNNPACK and not by NNPACK and QNNPACK.
    // So we do : return pthreadpool_get_threads_count_xnnpack(threadpool) as a hacky
    // solution for short term until unification.
    return reinterpret_cast<caffe2::ThreadPool*>(threadpool)->getNumThreads();
  }
  return 1;
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
