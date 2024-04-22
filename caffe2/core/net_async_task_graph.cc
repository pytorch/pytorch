#include "caffe2/core/net_async_task_graph.h"

#include "caffe2/core/net_parallel.h"

namespace caffe2 {

AsyncTaskGraph::AsyncTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options)
    : helper_(helper), options_(options), frozen_(false) {}

bool AsyncTaskGraph::CreateNode(
    int node_id,
    const std::vector<OperatorBase*>& ops) {
  CAFFE_ENFORCE(!frozen_);
  if (!nodes_.count(node_id)) {
    nodes_[node_id] = std::make_unique<AsyncTask>(ops);
    return true;
  } else {
    return false;
  }
}

bool AsyncTaskGraph::AddDependency(
    int child_node_id,
    const std::vector<int>& parent_node_ids) {
  CAFFE_ENFORCE(!frozen_);
  CAFFE_ENFORCE(!parent_node_ids.empty());
  CAFFE_ENFORCE(nodes_.count(child_node_id));
  for (auto node_id : parent_node_ids) {
    CAFFE_ENFORCE(nodes_.count(node_id));
  }
  CAFFE_ENFORCE(!parents_.count(child_node_id));

  auto* child_task = nodes_[child_node_id].get();
  auto child_device = child_task->GetDeviceOption();

  std::vector<AsyncTaskFuture*> parent_futures;
  for (auto node_id : parent_node_ids) {
    parents_[child_node_id].insert(node_id);
    children_[node_id].insert(child_node_id);
    parent_futures.push_back(&nodes_[node_id]->GetFuture());
  }

  AsyncTaskFuture* parents_future = nullptr;
  if (parent_futures.size() > 1) {
    edge_futures_.push_back(
        std::make_unique<AsyncTaskFuture>(parent_futures));
    parents_future = edge_futures_.back().get();
  } else {
    CAFFE_ENFORCE_EQ(parent_futures.size(), 1);
    parents_future = parent_futures.back();
  }

  // TODO: CUDA polling
  parents_future->SetCallback(
      [this, child_task, child_device](const AsyncTaskFuture* f) {
        CAFFE_ENFORCE(f->IsCompleted());
        if (!f->IsFailed()) {
          // if we're in the correct thread pool and DFS scheduling is enabled,
          // immediately call task inline, otherwise send task into thread pool
          auto* pool = helper_->GetPool(child_device);
          if (pool->inThreadPool() && options_.use_dfs_scheduling_) {
            child_task->Run(options_);
          } else {
            pool->run([this, child_task]() { child_task->Run(options_); });
          }
        } else {
          // skip task execution and propagate error further
          child_task->GetFuture().SetCompleted(f->ErrorMessage().c_str());
        }
      });

  return true;
}

void AsyncTaskGraph::FreezeGraph() {
  if (frozen_) {
    return;
  }

  CAFFE_ENFORCE(!run_future_);
  CAFFE_ENFORCE(root_tasks_.empty());

  std::vector<AsyncTaskFuture*> final_futures;
  for (auto& kv : nodes_) {
    auto task_id = kv.first;
    auto* task = kv.second.get();

    if (parents_[task_id].empty()) {
      root_tasks_.push_back(task);
    }

    if (children_[task_id].empty()) {
      auto& future = task->GetFuture();
      final_futures.push_back(&future);
    }
  }

  CAFFE_ENFORCE(!root_tasks_.empty());
  CAFFE_ENFORCE(!final_futures.empty());

  run_future_ = std::make_unique<AsyncTaskFuture>(final_futures);

  frozen_ = true;
}

AsyncTaskFuture* AsyncTaskGraph::ExecuteGraph() {
  CAFFE_ENFORCE(frozen_);
  CAFFE_ENFORCE(run_future_ && !run_future_->IsCompleted());

  // TODO: run root tasks inline in inference mode
  for (auto* task : root_tasks_) {
    auto task_device = task->GetDeviceOption();
    helper_->GetPool(task_device)->run([this, task]() { task->Run(options_); });
  }

  return run_future_.get();
}

AsyncTaskFuture* AsyncTaskGraph::GetFuture() {
  CAFFE_ENFORCE(frozen_);
  return run_future_.get();
}

void AsyncTaskGraph::Reset() {
  CAFFE_ENFORCE(frozen_);
  for (auto& kv : nodes_) {
    kv.second->Reset();
  }
  for (auto& future : edge_futures_) {
    future->ResetState();
  }
  if (run_future_) {
    run_future_->ResetState();
  }
}

}; // namespace caffe2
