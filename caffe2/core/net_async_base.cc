#include "caffe2/core/net_async_base.h"

#include "caffe2/core/net_async_tracing.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

// experimental support for multiple streams per worker per GPU
C10_DEFINE_int(
    caffe2_streams_per_gpu,
    1,
    "Number of streams per worker per GPU"
    " to use in GPU thread pool (experimental)");

C10_DECLARE_bool(caffe2_dag_net_collect_stats);

C10_DEFINE_bool(
    caffe2_net_async_finish_chain,
    false,
    "Wait for chain to finish");

C10_DEFINE_bool(
    caffe2_net_async_always_schedule_child,
    false,
    "Always schedule child chains from parent chain");

C10_DEFINE_int(
    caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor");

C10_DEFINE_int(
    caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor");

C10_DEFINE_int(
    caffe2_net_async_cpu_pool_size,
    0,
    "Number of threads in CPU pool by default");

C10_DEFINE_bool(
    caffe2_net_async_check_stream_status,
    false,
    "Select next non-busy stream");

C10_DEFINE_bool(
    caffe2_net_async_use_single_pool,
    false,
    "Use single thread pool for all devices");

C10_DEFINE_bool(
    caffe2_net_async_use_per_net_pools,
    false,
    "Use per net thread pools");

namespace caffe2 {

std::vector<int>& AsyncNetBase::getStreamCounters() {
  static thread_local std::vector<int> stream_counters_;
  return stream_counters_;
}

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

  execution_chains_ = dag_utils::computeChains(operator_nodes_);
  chains_.reserve(execution_chains_.size());
  for (const auto& kv : execution_chains_) {
    chains_.push_back(kv.second);
  }
  chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

  events_.reserve(chains_.size());
  for (const auto& chain : chains_) {
    const auto& last_op = operators_[chain.back()];
    events_.push_back(&last_op->event());
    for (const auto& op_id : chain) {
      if (op_id == chain.back() || op_id == chain.front()) {
        continue;
      }
      const auto& op = operators_[op_id];
      op->DisableEvent();
    }
  }

  num_workers_ = net_def->has_num_workers() ? net_def->num_workers() : -1;

  tracer_ = tracing::create(this, net_def->name());
  if (tracer_) {
    LOG(INFO) << "Tracing net: " << net_def->name();
  }

  computeExecutionModeFlags();
}

bool AsyncNetBase::handleRunError() {
#ifdef CAFFE2_USE_EXCEPTION_PTR
  std::unique_lock<std::mutex> exception_lock(exception_mutex_);
  if (caught_exception_) {
    std::rethrow_exception(caught_exception_);
  }
#endif // CAFFE2_USE_EXCEPTION_PTR
  return success_;
}

bool AsyncNetBase::RunAsync() {
  tracing::startIter(tracer_);
  reset();
  return DoRunAsync();
}

TaskThreadPool* AsyncNetBase::poolGetter(
    PoolsMap& pools,
    int device_type,
    int device_id,
    int pool_size) {
  std::unique_lock<std::mutex> pools_lock(pools_mutex_);
  auto pool = pools[device_id][pool_size];
  if (!pool) {
    pool = ThreadPoolRegistry()->Create(
        DeviceTypeName(device_type), device_id, pool_size, use_per_net_pools_);
    pools[device_id][pool_size] = pool;
  }
  return pool.get();
}

TaskThreadPool* AsyncNetBase::pool(const DeviceOption& device_option) {
  if (use_single_pool_) {
    return poolGetter(cpu_pools_, PROTO_CPU, -1, num_workers_);
  }
  static const std::unordered_set<int> cpu_types{
      PROTO_CPU,
      PROTO_MKLDNN,
      PROTO_IDEEP,
      PROTO_ONLY_FOR_TEST,
  };
  if (cpu_types.find(device_option.device_type()) != cpu_types.end()) {
    auto numa_node_id = -1;
    if (device_option.has_numa_node_id()) {
      numa_node_id = device_option.numa_node_id();
      CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
    }
    CAFFE_ENFORCE_LT(
        numa_node_id,
        c10::FLAGS_caffe2_net_async_max_numa_nodes,
        "Invalid NUMA node id: ",
        numa_node_id);
    return poolGetter(cpu_pools_, PROTO_CPU, numa_node_id, num_workers_);
  } else if (device_option.device_type() == PROTO_CUDA) {
    auto gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE(
        gpu_id >= 0 && gpu_id < c10::FLAGS_caffe2_net_async_max_gpus,
        "Invalid GPU id: " + caffe2::to_string(gpu_id));
    return poolGetter(gpu_pools_, PROTO_CUDA, gpu_id, num_workers_);
  } else {
    CAFFE_THROW(
        "Unsupported device type " +
        caffe2::to_string(device_option.device_type()));
  }
}

int AsyncNetBase::stream(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  int stream_id = 0;
  if (device_option.device_type() == PROTO_CUDA) {
    int gpu_id = device_option.cuda_gpu_id();
    CAFFE_ENFORCE_GE(gpu_id, 0, "Invalid gpu id: " + caffe2::to_string(gpu_id));
    if ((unsigned)gpu_id >= getStreamCounters().size()) {
      getStreamCounters().resize(gpu_id + 1, 0);
    }
    do {
      stream_id = getStreamCounters().at(gpu_id)++;
      getStreamCounters().at(gpu_id) %= streams_per_gpu_;
    } while (check_stream_status_ && !isStreamFree(task_id, stream_id));
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
    const std::vector<EventStatus>* status,
    bool* parent_failed) {
  auto first_child_op_id = chains_[task_id].front();
  for (auto parent_id : parents(task_id)) {
    auto last_parent_op_id = chains_[parent_id].back();
    EventStatus parent_status;
    if (status) {
      parent_status = status->at(parent_id);
    } else {
      parent_status = operators_[last_parent_op_id]->event().Query();
    }

    if (parent_status == EventStatus::EVENT_FAILED) {
      if (parent_failed) {
        *parent_failed = true;
      }
      return false;
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

bool AsyncNetBase::canSchedule(int parent_id, int child_id) {
  auto& parent_event = event(parent_id);
  auto first_child_op_id = chains_[child_id].front();
  auto* first_child_op = operators_[first_child_op_id];
  return Event::CanSchedule(
      parent_event.GetType(),
      parent_event.Query(),
      first_child_op->event().GetType(),
      first_child_op->SupportsAsyncScheduling());
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

int AsyncNetBase::getParentCount(int child_id) {
  auto& child_ops = chains_[child_id];
  auto& child_node = operator_nodes_[child_ops.front()];
  return child_node.runtime_parent_count_.load();
}

int AsyncNetBase::updateParentCount(int child_id) {
  auto& child_ops = chains_[child_id];
  auto& child_node = operator_nodes_[child_ops.front()];
  int parent_count = --child_node.runtime_parent_count_;
  CAFFE_ENFORCE_GE(parent_count, 0);
  return parent_count;
}

bool AsyncNetBase::testAndSetScheduled(int task_id) {
  auto& task_ops = chains_[task_id];
  auto& task_op_node = operator_nodes_[task_ops.front()];
  return !task_op_node.scheduled_.test_and_set();
}

int AsyncNetBase::numOps(int task_id) const {
  return chains_[task_id].size();
}

const OperatorBase* AsyncNetBase::firstTaskOp(int task_id) const {
  auto op_id = chains_[task_id].front();
  return operator_nodes_[op_id].operator_.get();
}

const OperatorBase* AsyncNetBase::lastTaskOp(int task_id) const {
  auto op_id = chains_[task_id].back();
  return operator_nodes_[op_id].operator_.get();
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

void AsyncNetBase::reset() {
  for (auto& op : GetOperators()) {
    op->ResetEvent();
  }
  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto& task_ops = chains_[task_id];
    auto& task_op_node = operator_nodes_[task_ops.front()];
    task_op_node.runtime_parent_count_ = parents(task_id).size();
    task_op_node.scheduled_.clear();
  }

  success_ = true;
#ifdef CAFFE2_USE_EXCEPTION_PTR
  std::unique_lock<std::mutex> exception_lock(exception_mutex_);
  caught_exception_ = nullptr;
#endif // CAFFE2_USE_EXCEPTION_PTR
}

void AsyncNetBase::storeExceptionPtr() {
#ifdef CAFFE2_USE_EXCEPTION_PTR
  std::unique_lock<std::mutex> exception_lock(exception_mutex_);
  if (!caught_exception_) {
    caught_exception_ = std::current_exception();
  }
#endif // CAFFE2_USE_EXCEPTION_PTR
}

void AsyncNetBase::setTaskErrorMessage(
    int task_id,
    const std::string& err_msg) {
  if (query(task_id) == EventStatus::EVENT_INITIALIZED) {
    event(task_id).SetFinished(err_msg.c_str());
  }
}

bool AsyncNetBase::run(int task_id, int stream_id) {
  OperatorBase* op = nullptr;
  try {
    // Optionally insert async wait ops,
    // skip when using --caffe2_net_async_finish_chain -
    // all parents are guaranteed to be finished
    if (!finish_chain_) {
      asyncWait(task_id, stream_id, parents(task_id));
    }
    for (auto& op_id : chains_[task_id]) {
      op = operators_[op_id];
      TRACE_EVENT(
          tracing::TRACE_OP,
          op_id,
          tracing::TRACE_TASK,
          task_id,
          tracing::TRACE_STREAM,
          stream_id);
      bool success = op->RunAsync(stream_id);
      if (!success) {
        auto err_msg = "Failed to execute an op: " +
            (op->has_debug_def() ? op->type() : " unknown");
        setTaskErrorMessage(task_id, err_msg);
        LOG(ERROR) << err_msg;
        return false;
      }
    }

    op = nullptr;
    if (finish_chain_) {
      operators_[chains_[task_id].back()]->event().Finish();
    }
  } catch (const std::exception& e) {
    storeExceptionPtr();
    std::string err_msg = e.what();
    if (op) {
      err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
    }
    setTaskErrorMessage(task_id, err_msg);
    LOG(ERROR) << err_msg;
    return false;
  } catch (...) {
    storeExceptionPtr();
    std::string err_msg = "Failed to execute task: unknown error";
    if (op) {
      err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
    }
    setTaskErrorMessage(task_id, err_msg);
    LOG(ERROR) << err_msg;
    return false;
  }

  return true;
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

C10_DEFINE_SHARED_REGISTRY(ThreadPoolRegistry, TaskThreadPool, int, int, bool);

C10_REGISTER_CREATOR(ThreadPoolRegistry, CPU, GetAsyncNetCPUThreadPool);

/* static */
std::shared_ptr<TaskThreadPool>
GetAsyncNetCPUThreadPool(int numa_node_id, int pool_size, bool create_new) {
  // Note: numa_node_id = -1 corresponds to no NUMA used
  static std::
      unordered_map<int, std::unordered_map<int, std::weak_ptr<TaskThreadPool>>>
          pools;
  static std::mutex pool_mutex;

  if (pool_size <= 0) {
    if (c10::FLAGS_caffe2_net_async_cpu_pool_size > 0) {
      pool_size = c10::FLAGS_caffe2_net_async_cpu_pool_size;
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

  if (create_new) {
    LOG(INFO) << "Created new CPU pool, size: " << pool_size
              << "; NUMA node id: " << numa_node_id;
    return std::make_shared<TaskThreadPool>(pool_size, numa_node_id);
  } else {
    std::lock_guard<std::mutex> lock(pool_mutex);

    auto shared_pool = pools[numa_node_id][pool_size].lock();
    if (!shared_pool) {
      LOG(INFO) << "Created shared CPU pool, size: " << pool_size
                << "; NUMA node id: " << numa_node_id;
      shared_pool = std::make_shared<TaskThreadPool>(pool_size, numa_node_id);
      pools[numa_node_id][pool_size] = shared_pool;
    }
    return shared_pool;
  }
}

void AsyncNetBase::computeExecutionModeFlags() {
  static const std::string kDag = "dag";
  static const std::string kProfDag = "prof_dag";
  static const std::string kAsyncDag = "async_dag";
  static const std::string kSimpleNet = "simple";

  std::string net_type;
  if (net_def_->has_type() && !net_def_->type().empty()) {
    net_type = net_def_->type();
  } else {
    net_type = kSimpleNet;
  }
  if (net_type == kDag || net_type == kProfDag) {
    streams_per_gpu_ = 1;
    finish_chain_ = true;
    always_schedule_child_ = true;
    check_stream_status_ = false;
    use_single_pool_ = true;
    use_per_net_pools_ = true;
    is_blocking_ = true;
  } else if (net_type == kAsyncDag) {
    streams_per_gpu_ = 1;
    finish_chain_ = false;
    always_schedule_child_ = true;
    check_stream_status_ = false;
    use_single_pool_ = true;
    use_per_net_pools_ = true;
    is_blocking_ = true;
  } else {
    streams_per_gpu_ = c10::FLAGS_caffe2_streams_per_gpu;
    finish_chain_ = c10::FLAGS_caffe2_net_async_finish_chain;
    always_schedule_child_ = c10::FLAGS_caffe2_net_async_always_schedule_child;
    check_stream_status_ = c10::FLAGS_caffe2_net_async_check_stream_status;
    use_single_pool_ = c10::FLAGS_caffe2_net_async_use_single_pool;
    use_per_net_pools_ = c10::FLAGS_caffe2_net_async_use_per_net_pools;
    is_blocking_ = false;
  }
}

} // namespace caffe2
