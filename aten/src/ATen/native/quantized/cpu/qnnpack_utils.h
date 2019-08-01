#pragma once

#ifdef USE_QNNPACK
#include <qnnpack.h>
#ifdef C10_MOBILE
#include "caffe2/utils/threadpool/ThreadPool.h"
#else
#include <thread>
#endif

struct QnnpackOperatorDeleter {
  void operator()(qnnp_operator_t op) {
    qnnp_delete_operator(op);
  }
};

class ThreadPoolMobile {
public:
  static pthreadpool_t qnnpack_threadpool();
#ifdef C10_MOBILE
private:
  static std::unique_ptr<caffe2::ThreadPool> thread_pool_;
  static std::mutex thread_pool_creation_mutex_;
#endif
};

#endif
