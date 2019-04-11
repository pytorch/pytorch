#ifndef CAFFE2_NET_TBB_TASK_GRAPH_H
#define CAFFE2_NET_TBB_TASK_GRAPH_H

#include "caffe2/core/net_async_base.h"
#include "caffe2/core/net_async_task_graph.h"
#include "caffe2/core/operator.h"

#include <tbb/flow_graph.h>
#include <tbb/task_arena.h>
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include <tbb/task_scheduler_observer.h>

namespace caffe2 {

class TBBTaskGraph : public AsyncTaskGraphBase {
 public:
  TBBTaskGraph(ExecutorHelper* helper, const ExecutionOptions& options);
  virtual ~TBBTaskGraph();

  bool CreateNode(int node_id, const std::vector<OperatorBase*>& ops) override;

  bool AddDependency(int child_node_id, const std::vector<int>& parent_node_ids)
      override;

  void FreezeGraph() override;

  AsyncTaskFuture* ExecuteGraph() override;

  AsyncTaskFuture* GetFuture() override;

  void Reset() override;

 private:
  ExecutorHelper* helper_;
  ExecutionOptions options_;

  bool frozen_;

  std::unordered_map<int, std::unique_ptr<AsyncTask>> nodes_;
  std::unordered_map<int, std::unordered_set<int>> parents_;
  std::unordered_map<int, std::unordered_set<int>> children_;

  std::unique_ptr<tbb::task_scheduler_init> scheduler_init_;
  std::unique_ptr<tbb::flow::continue_node<tbb::flow::continue_msg>> dag_root_,
      dag_exit_;
  std::vector<
      std::unique_ptr<tbb::flow::continue_node<tbb::flow::continue_msg>>>
      tbb_flow_nodes_;
  std::vector<std::unique_ptr<tbb::flow::graph_node>> async_edges_;
  std::vector<std::unique_ptr<tbb::task_scheduler_observer>>
      scheduler_observers_;

  std::vector<std::unique_ptr<tbb::flow::graph>> dags_;
  std::vector<std::unique_ptr<tbb::task_arena>> tbb_task_arena_;

  std::unique_ptr<AsyncTaskFuture> run_future_;
};

std::shared_ptr<AsyncTaskGraphBase> GetTBBTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options);

} // namespace caffe2

#endif // CAFFE2_NET_TBB_TASK_GRAPH_H
