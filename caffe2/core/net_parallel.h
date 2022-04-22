#ifndef CAFFE2_CORE_NET_PARALLEL_H
#define CAFFE2_CORE_NET_PARALLEL_H

#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_async_task_graph.h"

C10_DECLARE_string(caffe2_task_graph_engine);

namespace caffe2 {

class ParallelNetExecutorHelper;

class TORCH_API ParallelNet : public NetBase {
 public:
  ParallelNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);

  bool RunAsync() override;
  void Wait() override;

  bool SupportsAsync() override;
  std::vector<OperatorBase*> GetOperators() const override;

  TaskThreadPoolBase* Pool(const DeviceOption& device_option);

 protected:
  bool handleRunError() override;
  virtual void finishRun();
  virtual void reset();

  ExecutionOptions options_;
  int num_workers_;

  std::unique_ptr<ParallelNetExecutorHelper> helper_;
  std::shared_ptr<AsyncTaskGraphBase> task_graph_;
  AsyncTaskFuture* run_future_;

  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<OperatorBase*> operators_;

  std::mutex pools_mutex_;
  typedef std::unordered_map<
      int,
      std::unordered_map<int, std::shared_ptr<TaskThreadPoolBase>>>
      PoolsMap;
  PoolsMap cpu_pools_;
  PoolsMap gpu_pools_;
  TaskThreadPoolBase*
  poolGetter(PoolsMap& pools, int device_type, int device_id, int pool_size);

  friend class ParallelNetExecutorHelper;
  C10_DISABLE_COPY_AND_ASSIGN(ParallelNet);
};

C10_DECLARE_SHARED_REGISTRY(
    TaskGraphRegistry,
    AsyncTaskGraphBase,
    ExecutorHelper*,
    const ExecutionOptions&);

std::shared_ptr<AsyncTaskGraphBase> GetAsyncTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options);

class ParallelNetExecutorHelper : public ExecutorHelper {
 public:
  explicit ParallelNetExecutorHelper(ParallelNet* net) : net_(net) {}
  TaskThreadPoolBase* GetPool(const DeviceOption& option) const override {
    return net_->Pool(option);
  }

  std::vector<OperatorBase*> GetOperators() const override {
    return net_->GetOperators();
  }

  int GetNumWorkers() const override {
    return net_->num_workers_;
  }

 private:
  ParallelNet* net_;
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_PARALLEL_H
