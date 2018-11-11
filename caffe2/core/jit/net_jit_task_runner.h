#ifndef CAFFE2_NET_JIT_TASK_RUNNER_H
#define CAFFE2_NET_JIT_TASK_RUNNER_H

#include "caffe2/core/jit/net_jit_future.h"

namespace caffe2 {

class JITC2Task;

class JITC2TaskRunner {
 public:
  virtual JITFuture* RunTask(const std::shared_ptr<JITC2Task>& task) = 0;

  virtual ~JITC2TaskRunner() {}
};

} // namespace caffe2

#endif // CAFFE2_NET_JIT_TASK_RUNNER_H
