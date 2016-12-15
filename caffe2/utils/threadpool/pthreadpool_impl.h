#ifndef CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
#define CAFFE2_UTILS_PTHREADPOOL_IMPL_H_

#include "ThreadPoolCommon.h"

#ifndef CAFFE2_THREADPOOL_MOBILE
#error "mobile build state not defined"
#endif

#if CAFFE2_THREADPOOL_MOBILE

namespace caffe2 {

struct ThreadPool;

} // namespace caffe2

extern "C" {

// Wrapper for the caffe2 threadpool for the usage of NNPACK
struct pthreadpool {
  pthreadpool(caffe2::ThreadPool* pool) : pool_(pool) {}
  caffe2::ThreadPool* pool_;
};

} // extern "C"

#endif // CAFFE2_THREADPOOL_MOBILE

#endif  // CAFFE2_UTILS_PTHREADPOOL_IMPL_H_
