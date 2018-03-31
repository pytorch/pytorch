#ifndef CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
#define CAFFE2_UTILS_PTHREADPOOL_IMPL_H_

#include "ThreadPoolCommon.h"


namespace caffe2 {

class ThreadPool;

} // namespace caffe2

extern "C" {

// Wrapper for the caffe2 threadpool for the usage of NNPACK
struct pthreadpool {
  pthreadpool(caffe2::ThreadPool* pool) : pool_(pool) {}
  caffe2::ThreadPool* pool_;
};

} // extern "C"

#endif  // CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
