#include "caffe2/intra_op_parallel/net_tbb_task_graph.h"

#include "caffe2/core/net_parallel.h"

#include <tbb/task_scheduler_init.h>

using namespace tbb::flow;

namespace caffe2 {

namespace {

class pinning_observer : public tbb::task_scheduler_observer {
 public:
  pinning_observer(tbb::task_arena& arena, int numa_node_id)
      : tbb::task_scheduler_observer(arena), numa_node_id_(numa_node_id) {
    observe(true);
  } // activate the observer

  void on_scheduler_entry(bool /* unused */) override {
    NUMABind(numa_node_id_);
  }

 private:
  int numa_node_id_;
};

// Operators without explicit numa_node_id setting
// will be run at DEFAULT_NUMA_NODE_ID
static constexpr int DEFAULT_NUMA_NODE_ID = 0;

int get_numa_node_id(const DeviceOption& dev) {
  int numa_node_id = DEFAULT_NUMA_NODE_ID;
  if (IsCPUDeviceType(dev.device_type()) && dev.has_numa_node_id()) {
    numa_node_id = dev.numa_node_id();
  }
  return numa_node_id;
}

// This function can be used for creating a thin async_node at each
// cross-graph edge across different numa nodes.
// It prevents task bypassing which would violate affinity.
// receiver_g graph object must be the receiver's graph
template <typename T>
std::unique_ptr<graph_node>
make_async_edge(sender<T>& s, receiver<T>& r, graph& receiver_g) {
  typedef async_node<T, T> async_node_t;
  auto node = new async_node_t(
      receiver_g,
      unlimited,
      [](T msg, typename async_node_t::gateway_type& gw) { gw.try_put(msg); });
  make_edge(s, *node);
  make_edge(*node, r);
  return std::unique_ptr<graph_node>(node);
}

} // anonymous namespace

TBBTaskGraph::TBBTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options)
    : helper_(helper), options_(options), frozen_(false) {
  const auto& ops = helper_->GetOperators();
  int max_numa_node_id = 0;
  for (auto op_id = 0; op_id < ops.size(); ++op_id) {
    auto dev = ops.at(op_id)->device_option();
    int numa_node_id = get_numa_node_id(dev);
    max_numa_node_id = std::max(max_numa_node_id, numa_node_id);
  }

  LOG(INFO) << "# NUMA nodes: " << GetNumNUMANodes()
            << ". hardware concurrency: "
            << std::thread::hardware_concurrency();

  int num_workers = helper_->GetNumWorkers();
  scheduler_init_.reset(new tbb::task_scheduler_init(
      num_workers == -1 ? tbb::task_scheduler_init::automatic
                        : (max_numa_node_id + 1) * num_workers));

  dags_.resize(max_numa_node_id + 1);
  tbb_task_arena_.resize(max_numa_node_id + 1);
  scheduler_observers_.resize(max_numa_node_id + 1);

  for (int numa_node_id = max_numa_node_id; numa_node_id >= 0; --numa_node_id) {
    tbb_task_arena_[numa_node_id].reset(new tbb::task_arena(
        num_workers == -1 ? tbb::task_arena::automatic : num_workers, 0));

    scheduler_observers_[numa_node_id].reset(
        new pinning_observer(*tbb_task_arena_[numa_node_id], numa_node_id));
    dags_[numa_node_id].reset(new graph);
    // Attach the graph to the task arena
    tbb_task_arena_[numa_node_id]->execute(
        [this, numa_node_id]() { dags_[numa_node_id]->reset(); });
  }

  // To correctly wait for completion of all tasks in all graphs,
  // we artificially increase the wait counter for g1, and decrease it when
  // all continue_nodes complete.
  // For that, we add a dedicated start node which signals to all nodes
  // that have no predecessors, and a dedicated final node
  // which receives signals from all nodes that have no successor.
  dag_root_.reset(new continue_node<continue_msg>(
      *dags_[DEFAULT_NUMA_NODE_ID],
      [this](continue_msg) { dags_[DEFAULT_NUMA_NODE_ID]->reserve_wait(); }));
  run_future_ = caffe2::make_unique<AsyncTaskFuture>();
  dag_exit_.reset(new continue_node<continue_msg>(
      *dags_[DEFAULT_NUMA_NODE_ID], [this](continue_msg) {
        run_future_->SetCompleted();
        dags_[DEFAULT_NUMA_NODE_ID]->release_wait();
      }));
}

TBBTaskGraph::~TBBTaskGraph() {
  tbb_flow_nodes_.clear();
  async_edges_.clear();
  dag_root_.reset();
  dag_exit_.reset();
  // Need to clear dags_ last.
  dags_.clear();
}

bool TBBTaskGraph::CreateNode(
    int node_id,
    const std::vector<OperatorBase*>& ops) {
  CAFFE_ENFORCE(!frozen_);
  CAFFE_ENFORCE(!ops.empty());
  CAFFE_ENFORCE(!nodes_.count(node_id));

  nodes_[node_id] = caffe2::make_unique<AsyncTask>(ops);

  if (node_id >= tbb_flow_nodes_.size()) {
    tbb_flow_nodes_.resize(node_id + 1);
  }

  auto dev = ops.back()->device_option();
  int numa_node_id = get_numa_node_id(dev);
  tbb_flow_nodes_[node_id] = caffe2::make_unique<continue_node<continue_msg>>(
      *dags_[numa_node_id],
      [this, node_id](continue_msg) { nodes_[node_id]->Run(options_); });
  return true;
}

bool TBBTaskGraph::AddDependency(
    int child_id,
    const std::vector<int>& parent_node_ids) {
  CAFFE_ENFORCE(!frozen_);
  CAFFE_ENFORCE(nodes_.count(child_id));
  for (auto parent_id : parent_node_ids) {
    CAFFE_ENFORCE(nodes_.count(parent_id));
    parents_[child_id].insert(parent_id);
    children_[parent_id].insert(child_id);
  }

  const auto& dev = nodes_[child_id]->GetDeviceOption();
  int numa_node_id = get_numa_node_id(dev);

  for (auto parent_id : parent_node_ids) {
    const auto& parent_dev = nodes_[parent_id]->GetDeviceOption();
    int parent_numa_node_id = get_numa_node_id(parent_dev);
    if (parent_numa_node_id != numa_node_id) {
      async_edges_.push_back(make_async_edge(
          *tbb_flow_nodes_[parent_id],
          *tbb_flow_nodes_[child_id],
          *dags_[numa_node_id]));
    } else {
      make_edge(*tbb_flow_nodes_[parent_id], *tbb_flow_nodes_[child_id]);
    }
  }

  return true;
}

void TBBTaskGraph::FreezeGraph() {
  if (frozen_) {
    return;
  }

  for (auto& kv : nodes_) {
    int node_id = kv.first;
    const auto& dev = nodes_[node_id]->GetDeviceOption();
    int numa_node_id = get_numa_node_id(dev);

    if (parents_[node_id].empty()) {
      if (numa_node_id != DEFAULT_NUMA_NODE_ID) {
        async_edges_.push_back(make_async_edge(
            *dag_root_, *tbb_flow_nodes_[node_id], *dags_[numa_node_id]));
      } else {
        make_edge(*dag_root_, *tbb_flow_nodes_[node_id]);
      }
    }

    if (children_[node_id].empty()) {
      if (numa_node_id != DEFAULT_NUMA_NODE_ID) {
        async_edges_.push_back(make_async_edge(
            *tbb_flow_nodes_[node_id],
            *dag_exit_,
            *dags_[DEFAULT_NUMA_NODE_ID]));
      } else {
        make_edge(*tbb_flow_nodes_[node_id], *dag_exit_);
      }
    }
  }

  frozen_ = true;
}

AsyncTaskFuture* TBBTaskGraph::ExecuteGraph() {
  CAFFE_ENFORCE(frozen_);
  CAFFE_ENFORCE(run_future_ && !run_future_->IsCompleted());

  tbb_task_arena_[DEFAULT_NUMA_NODE_ID]->execute(
      [this] { dag_root_->try_put(continue_msg()); });
  // tbb_task_arena_[DEFAULT_NUMA_NODE_ID + 1]->execute(
  //    [this] { dags_[DEFAULT_NUMA_NODE_ID + 1]->wait_for_all(); });

  return run_future_.get();
}

AsyncTaskFuture* TBBTaskGraph::GetFuture() {
  CAFFE_ENFORCE(frozen_);
  return run_future_.get();
}

void TBBTaskGraph::Reset() {
  CAFFE_ENFORCE(frozen_);
  CAFFE_ENFORCE(run_future_);
  for (auto& kv : nodes_) {
    kv.second->Reset();
  }
  run_future_->ResetState();
}

std::shared_ptr<AsyncTaskGraphBase> GetTBBTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options) {
  return std::make_shared<TBBTaskGraph>(helper, options);
}

C10_REGISTER_CREATOR(TaskGraphRegistry, tbb, GetTBBTaskGraph);

}; // namespace caffe2
