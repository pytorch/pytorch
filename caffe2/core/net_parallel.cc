#include "caffe2/core/net_parallel.h"

#include "caffe2/core/operator.h"

#include <sstream>

C10_DEFINE_string(
    caffe2_task_graph_engine,
    "futures",
    "Task graph engine type used by net executor");

namespace caffe2 {

ParallelNet::ParallelNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws), options_(net_def), run_future_(nullptr) {
  num_workers_ = net_def->num_workers();
  CAFFE_ENFORCE_GT(
      num_workers_, 0, "Expected positive number of worker threads");

  helper_ = std::make_unique<ParallelNetExecutorHelper>(this);

  // initialize operators
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    auto op = node.operator_.get();
    op->SetExecutorHelper(helper_.get());
    operators_.push_back(op);
  }

  task_graph_ = TaskGraphRegistry()->Create(
      FLAGS_caffe2_task_graph_engine, helper_.get(), options_);
  CAFFE_ENFORCE(task_graph_, "Couldn't initialize task graph");

  // compute chains
  // TODO: inference mode for chaining
  auto execution_chains = dag_utils::computeChains(operator_nodes_);
  std::vector<std::vector<int>> chains;
  chains.reserve(execution_chains.size());
  for (const auto& kv : execution_chains) {
    chains.push_back(kv.second);
  }
  auto chain_nodes = dag_utils::prepareChainGraphNodes(operator_nodes_, chains);
  CAFFE_ENFORCE_EQ(chains.size(), chain_nodes.size());

  // disable unused events
  for (const auto& chain : chains) {
    for (const auto& op_id : chain) {
      if (op_id == chain.back() || op_id == chain.front()) {
        continue;
      }
      auto op = operators_[op_id];
      if (IsCPUDeviceType(op->device_option().device_type()) &&
          op->HasAsyncPart()) {
        continue;
      }
      op->DisableEvent();
    }
  }

  // initialize task graph
  for (auto chain_id = 0U; chain_id < chains.size(); ++chain_id) {
    std::vector<OperatorBase*> ops;
    ops.reserve(chains[chain_id].size());
    for (auto op_id : chains[chain_id]) {
      ops.push_back(operators_[op_id]);
    }
    CAFFE_ENFORCE(task_graph_->CreateNode(chain_id, ops));
  }
  for (auto chain_id = 0U; chain_id < chain_nodes.size(); ++chain_id) {
    if (!chain_nodes[chain_id].parents_.empty()) {
      CAFFE_ENFORCE(
          task_graph_->AddDependency(chain_id, chain_nodes[chain_id].parents_));
    }
  }

  // Freeze graph and initialize graph execution future
  task_graph_->FreezeGraph();
  run_future_ = task_graph_->GetFuture();
  run_future_->SetCallback([this](const AsyncTaskFuture* /* unused */) {
    StopAllObservers();
    finishRun();
  });

  LOG(INFO) << "Initialized parallel net: '" << Name()
            << "', #ops: " << net_def->op_size()
            << ", #chains: " << chains.size() << ", #workers: " << num_workers_
            << ", dfs scheduling: " << options_.use_dfs_scheduling_
            << ", task graph engine: " << FLAGS_caffe2_task_graph_engine;
}

bool ParallelNet::RunAsync() {
  reset();
  StartAllObservers();

  try {
    task_graph_->ExecuteGraph();
  } catch (const std::exception&) {
    StopAllObservers();
    return false;
  }

  return true;
}

void ParallelNet::Wait() {
  CAFFE_ENFORCE(run_future_);
  run_future_->Wait();
}

void ParallelNet::reset() {
  task_graph_->Reset();
}

bool ParallelNet::handleRunError() {
  CAFFE_ENFORCE(run_future_ && run_future_->IsCompleted());
  // TODO: throw saved exceptions
  if (run_future_->IsFailed()) {
    LOG(ERROR) << "Failed parallel run (" << Name()
               << "): " << run_future_->ErrorMessage();
  }
  return !run_future_->IsFailed();
}

TaskThreadPoolBase* ParallelNet::poolGetter(
    PoolsMap& pools,
    int device_type,
    int device_id,
    int pool_size) {
  std::unique_lock<std::mutex> pools_lock(pools_mutex_);
  auto pool = pools[device_id][pool_size];
  if (!pool) {
    pool = c10::ThreadPoolRegistry()->Create(
        DeviceTypeName(device_type),
        device_id,
        pool_size,
        options_.use_per_net_pools_);
    pools[device_id][pool_size] = pool;
  }
  return pool.get();
}

TaskThreadPoolBase* ParallelNet::Pool(const DeviceOption& device_option) {
  if (options_.use_single_pool_) {
    return poolGetter(cpu_pools_, PROTO_CPU, -1, num_workers_);
  }
  const auto device_type = device_option.device_type();
  if (IsCPUDeviceType(device_type)) {
    auto numa_node_id = -1;
    if (device_option.has_numa_node_id()) {
      numa_node_id = device_option.numa_node_id();
      CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
    }
    CAFFE_ENFORCE_LT(
        numa_node_id,
        FLAGS_caffe2_net_async_max_numa_nodes,
        "Invalid NUMA node id: ",
        numa_node_id);
    return poolGetter(cpu_pools_, device_type, numa_node_id, num_workers_);
  } else if (IsGPUDeviceType(device_type)) {
    auto gpu_id = device_option.device_id();
    CAFFE_ENFORCE(
        gpu_id >= 0 && gpu_id < FLAGS_caffe2_net_async_max_gpus,
        "Invalid GPU id: " + caffe2::to_string(gpu_id));
    return poolGetter(gpu_pools_, device_type, gpu_id, num_workers_);
  } else {
    CAFFE_THROW("Unsupported device type " + caffe2::to_string(device_type));
  }
}

bool ParallelNet::SupportsAsync() {
  return true;
}

void ParallelNet::finishRun() {}

std::vector<OperatorBase*> ParallelNet::GetOperators() const {
  return operators_;
}

std::shared_ptr<AsyncTaskGraphBase> GetAsyncTaskGraph(
    ExecutorHelper* helper,
    const ExecutionOptions& options) {
  return std::make_shared<AsyncTaskGraph>(helper, options);
}

C10_DEFINE_SHARED_REGISTRY(
    TaskGraphRegistry,
    AsyncTaskGraphBase,
    ExecutorHelper*,
    const ExecutionOptions&);

C10_REGISTER_CREATOR(TaskGraphRegistry, futures, GetAsyncTaskGraph);

REGISTER_NET(parallel, ParallelNet);

} // namespace caffe2
