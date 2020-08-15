#ifndef CAFFE2_NET_ASYNC_TASK_GRAPH_H
#define CAFFE2_NET_ASYNC_TASK_GRAPH_H

#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_async_task.h"
#include "caffe2/core/net_async_task_future.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// AsyncTaskGraph represents an execution of a net, it owns the tasks and
// associated futures, sets up future callbacks and propagates errors.
// Usage steps:
// - Adding graph nodes and edges through CreateNode/AddDependency;
// - Freezing the graph (FreezeGraph), after the freezing a future
//   can be obtained using GetFuture;
// - Execution of the graph is scheduled through ExecuteGraph, after each
//   execution Reset must be called to prepare the graph for the next run

class AsyncTaskGraphBase {
 public:
  virtual bool CreateNode(
      int node_id,
      const std::vector<OperatorBase*>& ops) = 0;

  virtual bool AddDependency(
      int child_node_id,
      const std::vector<int>& parent_node_ids) = 0;

  virtual void FreezeGraph() = 0;

  virtual AsyncTaskFuture* ExecuteGraph() = 0;

  virtual AsyncTaskFuture* GetFuture() = 0;

  virtual void Reset() = 0;

  virtual ~AsyncTaskGraphBase() noexcept {}
};

class AsyncTaskGraph : public AsyncTaskGraphBase {
 public:
  AsyncTaskGraph(ExecutorHelper* helper, const ExecutionOptions& options);

  bool CreateNode(int node_id, const std::vector<OperatorBase*>& ops) override;

  bool AddDependency(int child_node_id, const std::vector<int>& parent_node_ids)
      override;

  void FreezeGraph() override;

  AsyncTaskFuture* ExecuteGraph() override;

  AsyncTaskFuture* GetFuture() override;

  void Reset() override;

 private:
  // used to, e.g., get access to executor's thread pools
  // TODO: pass tracer and counters through ExecutorHelper
  ExecutorHelper* helper_;
  ExecutionOptions options_;

  bool frozen_;

  std::unordered_map<int, std::unique_ptr<AsyncTask>> nodes_;
  std::unordered_map<int, std::unordered_set<int>> parents_;
  std::unordered_map<int, std::unordered_set<int>> children_;
  std::vector<std::unique_ptr<AsyncTaskFuture>> edge_futures_;

  std::vector<AsyncTask*> root_tasks_;

  std::unique_ptr<AsyncTaskFuture> run_future_;
};

} // namespace caffe2

#endif // CAFFE2_NET_ASYNC_TASK_GRAPH_H
