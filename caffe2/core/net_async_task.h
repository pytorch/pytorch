#ifndef CAFFE2_NET_ASYNC_TASK_H
#define CAFFE2_NET_ASYNC_TASK_H

#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_async_task_future.h"
#include "caffe2/core/operator.h"

#include <vector>

namespace caffe2 {

// AsyncTask represents an asynchronous execution of a chain of ops.
class AsyncTask {
 public:
  AsyncTask(const std::vector<OperatorBase*>& ops);

  bool Run(const ExecutionOptions& options);

  void Reset();

  DeviceOption GetDeviceOption() const;

  AsyncTaskFuture& GetFuture();
  const AsyncTaskFuture& GetFuture() const;

 private:
  void handleChainError(
      OperatorBase* op,
      const char* err_msg,
      bool save_exception = false);

  std::vector<OperatorBase*> ops_;
  DeviceOption device_option_;
  AsyncTaskFuture future_;
};

} // namespace caffe2

#endif // CAFFE2_NET_ASYNC_TASK_H
