#include "caffe2/utils/threadpool/pthreadpool.h"
#include "caffe2/utils/threadpool/pthreadpool_impl.h"
#include "caffe2/utils/threadpool/ThreadPool.h"


//
// External API
//

void pthreadpool_compute_1d(struct pthreadpool* threadpool,
                            pthreadpool_function_1d_t function,
                            void* argument,
                            size_t range) {
    threadpool->pool_->run(
      [function, argument](int threadId, size_t workId) {
        function(argument, workId);
      },
      range);
}

size_t pthreadpool_get_threads_count(struct pthreadpool* threadpool) {
  return threadpool->pool_->getNumThreads();
}
