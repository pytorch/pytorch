#include "caffe2/utils/thread_name.h"

#include <algorithm>

#if defined(__GLIBC__) && !defined(__APPLE__) && !defined(__ANDROID__)
#define CAFFE2_HAS_PTHREAD_SETNAME_NP
#endif

#ifdef CAFFE2_HAS_PTHREAD_SETNAME_NP
#include <pthread.h>
#endif

namespace caffe2 {

void setThreadName(std::string name) {
#ifdef CAFFE2_HAS_PTHREAD_SETNAME_NP
  constexpr size_t kMaxThreadName = 15;
  name.resize(std::min(name.size(), kMaxThreadName));

  pthread_setname_np(pthread_self(), name.c_str());
#endif
}

} // namespace caffe2
