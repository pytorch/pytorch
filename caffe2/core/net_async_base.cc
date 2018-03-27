#include "caffe2/core/net_async_polling.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

CAFFE2_DEFINE_int(
    caffe2_streams_per_gpu,
    32,
    "Number of streams per GPU to use in GPU thread pool");

CAFFE2_DECLARE_bool(caffe2_dag_net_collect_stats);

CAFFE2_DEFINE_bool(
    caffe2_net_async_finish_chain,
    false,
    "Wait for chain to finish");

CAFFE2_DEFINE_int(
    caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor");

CAFFE2_DEFINE_int(
    caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor");

CAFFE2_DEFINE_int(
    caffe2_net_async_cpu_pool_size,
    0,
    "Number of threads in CPU pool (default - number of cores)");

CAFFE2_DEFINE_bool(
    caffe2_net_async_check_stream_status,
    true,
    "Select next non-busy stream");

namespace caffe2 {

thread_local std::vector<int> AsyncNetBase::stream_counters_;

AsyncNetBase::AsyncNetBase(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    operators_.push_back(node.operator_.get());
  }

  const auto& execution_chains = dag_utils::computeChains(operator_nodes_);
  chains_.reserve(execution_chains.size());
  for (const auto& kv : execution_chains) {
    chains_.push_back(kv.second);
  }
  chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

  events_.reserve(chains_.size());
  for (const auto& chain : chains_) {
    const auto& op = operators_[chain.back()];
    events_.push_back(&op->event());
  }

  gpu_pools_.resize(FLAGS_caffe2_net_async_max_gpus);
  cpu_pools_.resize(FLAGS_caffe2_net_async_max_numa_nodes);
  DeviceOption cpu_option;
  cpu_option.set_device_type(CPU);
  cpu_pool_ = ThreadPoolRegistry()->Create(
      DeviceTypeName(cpu_option.device_type()), cpu_option);
}

std::shared_ptr<TaskThreadPool> AsyncNetBase::pool_getter(
    std::vector<std::shared_ptr<TaskThreadPool>>& pools,
    int pool_idx,
    const DeviceOption& device_option) {
  std::unique_lock<std::mutex> pools_lock(pools_mutex_);
  auto pool = pools[pool_idx];
  if (!pool) {
    pool = ThreadPoolRegistry()->Create(
        DeviceTypeName(device_option.device_type()), device_option);
    pools[pool_idx] = pool;
  }
  return pool;
}

std::shared_ptr<TaskThreadPool> AsyncNetBase::pool(
    const DeviceOption& device_option) {
  if (device_option.device_type() == CPU) {
    auto numa_node_id = device_option.numa_node_id();
    if (numa_node_id == -1) {
      return cpu_pool_;
    } else {
      CAFFE_ENFORCE(
          numa_node_id >= 0 &&
              numa_node_id < FLAGS_caffe2_net_async_max_numa_nodes,
          "Invalid NUMA node id: " + caffe2::to_string(numa_node_id));
      return pool_getter(cpu_pools_, numa_node_id, device_option);
    }
  } else if (device_option.device_type() == CUDA) {
    auto gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE(
        gpu_id >= 0 && gpu_id < FLAGS_caffe2_net_async_max_gpus,
        "Invalid GPU id: " + caffe2::to_string(gpu_id));
    return pool_getter(gpu_pools_, gpu_id, device_option);
  } else {
    CAFFE_THROW(
        "Unsupported device type " +
        caffe2::to_string(device_option.device_type()));
  }
}

int AsyncNetBase::stream(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  int stream_id = 0;
  if (device_option.device_type() == CUDA) {
    int gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE_GE(gpu_id, 0, "Invalid gpu id: " + caffe2::to_string(gpu_id));
    if (gpu_id >= stream_counters_.size()) {
      stream_counters_.resize(gpu_id + 1, 0);
    }
    do {
      stream_id = stream_counters_[gpu_id]++;
      stream_counters_[gpu_id] %= FLAGS_caffe2_streams_per_gpu;
    } while (!isStreamFree(task_id, stream_id) &&
             FLAGS_caffe2_net_async_check_stream_status);
  }
  return stream_id;
}

bool AsyncNetBase::isStreamFree(int task_id, int stream_id) const {
  auto& task = chains_[task_id];
  auto& last_task_op = operators_[task.back()];
  return last_task_op->IsStreamFree(stream_id);
}

bool AsyncNetBase::canSchedule(
    int task_id,
    const std::vector<EventStatus>* status) {
  auto first_child_op_id = chains_[task_id].front();
  for (auto parent_id : parents(task_id)) {
    auto last_parent_op_id = chains_[parent_id].back();
    EventStatus parent_status;
    if (status) {
      parent_status = status->at(parent_id);
    } else {
      parent_status = operators_[last_parent_op_id]->event().Query();
    }
    bool can_schedule = Event::CanSchedule(
        operators_[last_parent_op_id]->event().GetType(),
        parent_status,
        operators_[first_child_op_id]->event().GetType(),
        operators_[first_child_op_id]->SupportsAsyncScheduling());
    if (!can_schedule) {
      return false;
    }
  }

  return true;
}

int AsyncNetBase::tasksNum() const {
  return chains_.size();
}

Event& AsyncNetBase::event(int task_id) const {
  auto& task = chains_[task_id];
  auto& last_task_op = operators_[task.back()];
  return last_task_op->event();
}

EventStatus AsyncNetBase::query(int task_id) const {
  return event(task_id).Query();
}

const std::vector<int>& AsyncNetBase::children(int task_id) const {
  const auto& task_node = chain_nodes_[task_id];
  return task_node.children_;
}

const std::vector<int>& AsyncNetBase::parents(int task_id) const {
  const auto& task_node = chain_nodes_[task_id];
  return task_node.parents_;
}

void AsyncNetBase::asyncWait(
    int task_id,
    int stream_id,
    const std::vector<int>& wait_task_ids) const {
  auto first_op_id = chains_[task_id].front();
  auto& first_op = operators_[first_op_id];
  std::vector<const Event*> events;
  events.reserve(wait_task_ids.size());
  for (auto wait_task_id : wait_task_ids) {
    events.push_back(&event(wait_task_id));
  }
  first_op->WaitEvents(events, stream_id);
}

void AsyncNetBase::run(int task_id, int stream_id) {
  std::string err_msg;
  for (auto& op_id : chains_[task_id]) {
    auto& op = operators_[op_id];
    try {
      CAFFE_ENFORCE(op->RunAsync(stream_id), "Failed to execute an op");
    } catch (const std::exception& e) {
      CAFFE_THROW(
          std::string(e.what()) + ",  op " +
          (op->has_debug_def() ? op->type() : " unknown"));
    } catch (...) {
      CAFFE_THROW(
          "Failed to execute task: unknown error,  op " +
          (op->has_debug_def() ? op->type() : " unknown"));
    }
  }

  if (FLAGS_caffe2_net_async_finish_chain) {
    operators_[chains_[task_id].back()]->event().Finish();
  }
}

void AsyncNetBase::finishTasks(const std::unordered_set<int>& task_ids) {
  for (const auto& task_id : task_ids) {
    event(task_id).Finish();
  }
}

void AsyncNetBase::finalizeEvents() {
  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto status = query(task_id);
    if (status == EventStatus::EVENT_SCHEDULED) {
      event(task_id).Finish();
    } else if (status == EventStatus::EVENT_INITIALIZED) {
      event(task_id).SetFinished();
    }
  }
}

AsyncNetBase::~AsyncNetBase() {}

CAFFE_DEFINE_SHARED_REGISTRY(
    ThreadPoolRegistry,
    TaskThreadPool,
    const DeviceOption&);

namespace {
std::shared_ptr<TaskThreadPool> AsyncNetCPUThreadPoolCreator(
    const DeviceOption& device_option) {
  CAFFE_ENFORCE_EQ(
      device_option.device_type(),
      CPU,
      "Unexpected device type for CPU thread pool");
  return GetAsyncNetCPUThreadPool(device_option.numa_node_id());
}
} // namespace

CAFFE_REGISTER_CREATOR(ThreadPoolRegistry, CPU, AsyncNetCPUThreadPoolCreator);

/* static */
std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool(int numa_node_id) {
  // Note: numa_node_id = -1 (DeviceOption's default value) corresponds to
  // no NUMA used
  static std::unordered_map<int, std::weak_ptr<TaskThreadPool>> pools;
  static std::mutex pool_mutex;
  std::lock_guard<std::mutex> lock(pool_mutex);

  std::shared_ptr<TaskThreadPool> shared_pool = nullptr;
  if (pools.count(numa_node_id)) {
    shared_pool = pools.at(numa_node_id).lock();
  }
  if (!shared_pool) {
    auto pool_size = FLAGS_caffe2_net_async_cpu_pool_size;
    if (pool_size <= 0) {
      auto num_cores = std::thread::hardware_concurrency();
      CAFFE_ENFORCE(num_cores > 0, "Failed to get number of CPU cores");
      pool_size = num_cores;
    }
    LOG(INFO) << "Using cpu pool size: " << pool_size;
    shared_pool = std::make_shared<TaskThreadPool>(pool_size, numa_node_id);
    pools[numa_node_id] = shared_pool;
  }
  return shared_pool;
}

} // namespace caffe2
