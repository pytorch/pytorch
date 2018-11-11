#ifndef CAFFE2_NET_JIT_EXECUTOR_H
#define CAFFE2_NET_JIT_EXECUTOR_H

#include "caffe2/core/jit/net_jit.h"
#include "caffe2/core/jit/net_jit_future.h"
#include "caffe2/core/jit/net_jit_task.h"
#include "caffe2/core/jit/net_jit_task_runner.h"
#include "caffe2/core/net.h"

namespace caffe2 {

class JITExecutor : public NetBase, public JITC2TaskRunner {
 public:
  JITExecutor(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  ~JITExecutor() override;

  bool RunAsync() override;

  JITFuture* RunTask(const std::shared_ptr<JITC2Task>& task) override;

  void Wait() override;

  std::vector<OperatorBase*> GetOperators() const override {
    return jit_->GetC2Ops();
  }

  bool SupportsAsync() override {
    return true;
  }

  TaskThreadPoolBase* Pool(const DeviceOption& device_option);

  bool handleRunError() override;

 protected:
  virtual void reset();
  virtual void finishRun() {}
  void checkNetArguments();

  std::shared_ptr<JITC2Program> jit_;
  JITFuture* run_future_;
  int num_workers_;
  bool use_dfs_scheduling_;

 private:
  std::mutex pools_mutex_;
  typedef std::unordered_map<
      int,
      std::unordered_map<int, std::shared_ptr<TaskThreadPoolBase>>>
      PoolsMap;
  PoolsMap cpu_pools_;
  PoolsMap gpu_pools_;
  TaskThreadPoolBase*
  poolGetter(PoolsMap& pools, int device_type, int device_id, int pool_size);

  std::mutex tasks_mutex_;
  std::vector<std::shared_ptr<JITC2Task>> tasks_;

  C10_DISABLE_COPY_AND_ASSIGN(JITExecutor);
};

} // namespace caffe2

#endif // CAFFE2_NET_JIT_EXECUTOR_H
