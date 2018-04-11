#include "caffe2/core/net_async_polling.h"

#include "caffe2/core/net_async_tracing.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"

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
    "Number of threads in CPU pool by default");

CAFFE2_DEFINE_bool(
    caffe2_net_async_check_stream_status,
    true,
    "Select next non-busy stream");

CAFFE2_DEFINE_string(
    caffe2_net_async_tracing_filepath,
    "/tmp",
    "Path to save tracing information");

CAFFE2_DEFINE_string(
    caffe2_net_async_names_to_trace,
    "",
    "Comma-separated list of net names to trace");

CAFFE2_DEFINE_int(caffe2_net_async_tracing_nth, 100, "Trace every Nth batch");

namespace caffe2 {

thread_local std::vector<int> AsyncNetBase::stream_counters_;

AsyncNetBase::AsyncNetBase(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  helper_ = caffe2::make_unique<AsyncNetExecutorHelper>(this);
  operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    auto op_ptr = node.operator_.get();
    op_ptr->SetExecutorHelper(helper_.get());
    operators_.push_back(op_ptr);
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

  num_workers_ = net_def->has_num_workers() ? net_def->num_workers() : -1;
  batch_iter_ = 0;

  initTracer(net_def);
}

void AsyncNetBase::initTracer(const std::shared_ptr<const NetDef>& net_def) {
  auto tracing_nets = caffe2::split(',', FLAGS_caffe2_net_async_names_to_trace);
  trace_net_ = !net_def->name().empty() &&
      std::find(tracing_nets.begin(), tracing_nets.end(), net_def->name()) !=
          tracing_nets.end();
  trace_batch_ = false;
  timer_.Start();
  if (trace_net_) {
    auto fn = net_def->name();
    std::replace(fn.begin(), fn.end(), '/', '_');
    tracer_ = caffe2::make_unique<tracing::Tracer>();
    tracer_->init(
        this, FLAGS_caffe2_net_async_tracing_filepath + "/" + fn + ".json");
  }
}

bool AsyncNetBase::RunAsync() {
  trace_batch_ =
      trace_net_ && (++batch_iter_ % FLAGS_caffe2_net_async_tracing_nth == 0);
  for (auto& op : GetOperators()) {
    op->ResetEvent();
  }
  return DoRunAsync();
}

std::shared_ptr<TaskThreadPool> AsyncNetBase::pool_getter(
    PoolsMap& pools,
    int device_type,
    int device_id,
    int pool_size) {
  std::unique_lock<std::mutex> pools_lock(pools_mutex_);
  auto pool = pools[device_id][pool_size];
  if (!pool) {
    pool = ThreadPoolRegistry()->Create(
        DeviceTypeName(device_type), device_id, pool_size);
    pools[device_id][pool_size] = pool;
  }
  return pool;
}

std::shared_ptr<TaskThreadPool> AsyncNetBase::pool(
    const DeviceOption& device_option) {
  if (device_option.device_type() == CPU) {
    auto numa_node_id = device_option.numa_node_id();
    CAFFE_ENFORCE(
        numa_node_id >= -1 &&
            numa_node_id < FLAGS_caffe2_net_async_max_numa_nodes,
        "Invalid NUMA node id: " + caffe2::to_string(numa_node_id));
    return pool_getter(cpu_pools_, CPU, numa_node_id, num_workers_);
  } else if (device_option.device_type() == CUDA) {
    auto gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE(
        gpu_id >= 0 && gpu_id < FLAGS_caffe2_net_async_max_gpus,
        "Invalid GPU id: " + caffe2::to_string(gpu_id));
    return pool_getter(gpu_pools_, CUDA, gpu_id, num_workers_);
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

int AsyncNetBase::num_ops(int task_id) const {
  return chains_[task_id].size();
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

OperatorBase* AsyncNetBase::op(int op_idx) const {
  return operators_[op_idx];
}

void AsyncNetBase::run(int task_id, int stream_id) {
  std::string err_msg;
  for (auto& op_id : chains_[task_id]) {
    auto& op = operators_[op_id];
    try {
      TRACE_EVENT(
          tracing::TRACE_OP,
          op_id,
          tracing::TRACE_TASK,
          task_id,
          tracing::TRACE_STREAM,
          stream_id);
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

CAFFE_DEFINE_SHARED_REGISTRY(ThreadPoolRegistry, TaskThreadPool, int, int);

CAFFE_REGISTER_CREATOR(ThreadPoolRegistry, CPU, GetAsyncNetCPUThreadPool);

/* static */
std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool(
    int numa_node_id,
    int pool_size) {
  // Note: numa_node_id = -1 (DeviceOption's default value) corresponds to
  // no NUMA used
  static std::
      unordered_map<int, std::unordered_map<int, std::weak_ptr<TaskThreadPool>>>
          pools;
  static std::mutex pool_mutex;
  std::lock_guard<std::mutex> lock(pool_mutex);

  if (pool_size <= 0) {
    if (FLAGS_caffe2_net_async_cpu_pool_size > 0) {
      pool_size = FLAGS_caffe2_net_async_cpu_pool_size;
      LOG(INFO) << "Using default CPU pool size: " << pool_size
                << "; NUMA node id: " << numa_node_id;
    } else {
      auto num_cores = std::thread::hardware_concurrency();
      CAFFE_ENFORCE(num_cores > 0, "Failed to get number of CPU cores");
      LOG(INFO) << "Using estimated CPU pool size: " << num_cores
                << "; NUMA node id: " << numa_node_id;
      pool_size = num_cores;
    }
  } else {
    LOG(INFO) << "Using specified CPU pool size: " << pool_size
              << "; NUMA node id: " << numa_node_id;
  }

  auto shared_pool = pools[numa_node_id][pool_size].lock();
  if (!shared_pool) {
    LOG(INFO) << "Created CPU pool, size: " << pool_size
              << "; NUMA node id: " << numa_node_id;
    shared_pool = std::make_shared<TaskThreadPool>(pool_size, numa_node_id);
    pools[numa_node_id][pool_size] = shared_pool;
  }
  return shared_pool;
}

} // namespace caffe2
