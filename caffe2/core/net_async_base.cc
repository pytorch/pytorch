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

C10_DEFINE_bool(
    caffe2_net_async_inference_mode,
    false,
    "If set, use one single chain containing all ops");

C10_DEFINE_bool(
    caffe2_net_async_profile_operators,
    false,
    "If set, profile operators of the net regardless of net being prof_dag.");

C10_DEFINE_int(
    caffe2_net_async_max_gpus,
    16,
    "Max number of GPUs allowed in net async executor");

C10_DEFINE_int(
    caffe2_net_async_max_numa_nodes,
    8,
    "Max number of NUMA nodes allowed in net async executor");

C10_DEFINE_int(
    caffe2_net_async_thread_pool_size,
    0,
    "Number of threads in device thread pool by default");

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

C10_DEFINE_bool(
    caffe2_net_async_run_root_tasks_inline,
    false,
    "Run root tasks in current thread instread of scheduling to threadpool");

namespace caffe2 {

std::vector<int>& AsyncNetBase::getStreamCounters() {
  static thread_local std::vector<int> stream_counters_;
  return stream_counters_;
}

AsyncNetBase::AsyncNetBase(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws), options_(net_def), counters_(net_def) {
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  helper_ = std::make_unique<AsyncNetExecutorHelper>(this);
  operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    auto op_ptr = node.operator_.get();
    op_ptr->SetExecutorHelper(helper_.get());
    operators_.push_back(op_ptr);
  }

  if (FLAGS_caffe2_net_async_inference_mode) {
    execution_chains_ = dag_utils::computeGroups(operator_nodes_);
  } else {
    execution_chains_ = dag_utils::computeChains(operator_nodes_);
  }
  chains_.reserve(execution_chains_.size());
  for (const auto& kv : execution_chains_) {
    chains_.push_back(kv.second);
  }
  chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

  events_.reserve(chains_.size());
  for (const auto& chain : chains_) {
    const auto& last_op = operators_[chain.back()];
    events_.push_back(&last_op->event());
    // keep events for inner chain ops in case of profiling
    if (!options_.report_stats_) {
      for (const auto& op_id : chain) {
        if (op_id == chain.back() || op_id == chain.front()) {
          continue;
        }
        const auto& op = operators_[op_id];
        op->DisableEvent();
      }
    }
  }

  num_workers_ = net_def->has_num_workers() ? net_def->num_workers() : -1;

  tracer_ = tracing::create(this, net_def->name());
  if (tracer_) {
    LOG(INFO) << "Tracing net: " << net_def->name();
  }
}

bool AsyncNetBase::handleRunError() {
#ifdef CAFFE2_USE_EXCEPTION_PTR
  // Check net's events for exceptions and rethrow chronologically the first one
  int first_exc_task_id = -1;
  int64_t first_exc_ts = 0;
  for (int task_id = 0; task_id < tasksNum(); ++task_id) {
    if (event(task_id).HasException()) {
      if (first_exc_task_id >= 0) {
        auto exc_ts = event(task_id).ExceptionTimestamp();
        if (exc_ts < first_exc_ts) {
          first_exc_task_id = task_id;
          first_exc_ts = exc_ts;
        }
      } else {
        first_exc_task_id = task_id;
        first_exc_ts = event(task_id).ExceptionTimestamp();
      }
    }
  }
  if (first_exc_task_id >= 0) {
    LOG(ERROR) << "Rethrowing exception from the run of '" << Name() << "'";
    event(first_exc_task_id).RethrowException();
  }
#endif // CAFFE2_USE_EXCEPTION_PTR

  if (!success_) {
    LOG(ERROR) << "Error encountered in the run of '" << Name() << "'";
  }
  return success_;
}

bool AsyncNetBase::RunAsync() {
  tracing::startIter(tracer_);
  reset();
  return DoRunAsync();
}

TaskThreadPoolBase* AsyncNetBase::poolGetter(
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

TaskThreadPoolBase* AsyncNetBase::pool() {
  // By default using a non-pinned CPU option
  DeviceOption dev;
  dev.set_device_type(PROTO_CPU);
  return pool(dev);
}

TaskThreadPoolBase* AsyncNetBase::pool(const DeviceOption& device_option) {
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
        "Invalid GPU id: " + c10::to_string(gpu_id));
    return poolGetter(gpu_pools_, device_type, gpu_id, num_workers_);
  } else {
    CAFFE_THROW("Unsupported device type " + c10::to_string(device_type));
  }
}

int AsyncNetBase::stream(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  int stream_id = 0;
  if (IsGPUDeviceType(device_option.device_type())) {
    int gpu_id = device_option.device_id();
    CAFFE_ENFORCE_GE(gpu_id, 0, "Invalid gpu id: " + c10::to_string(gpu_id));
    if ((unsigned)gpu_id >= getStreamCounters().size()) {
      getStreamCounters().resize(gpu_id + 1, 0);
    }
    do {
      stream_id = getStreamCounters().at(gpu_id)++;
      getStreamCounters().at(gpu_id) %= options_.streams_per_gpu_;
    } while (options_.check_stream_status_ &&
             !isStreamFree(task_id, stream_id));
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

int AsyncNetBase::firstTaskOpId(int task_id) const {
  return chains_[task_id].front();
}

int AsyncNetBase::lastTaskOpId(int task_id) const {
  return chains_[task_id].back();
}

const OperatorBase* AsyncNetBase::firstTaskOp(int task_id) const {
  return operator_nodes_[firstTaskOpId(task_id)].operator_.get();
}

const OperatorBase* AsyncNetBase::lastTaskOp(int task_id) const {
  return operator_nodes_[lastTaskOpId(task_id)].operator_.get();
}

OperatorBase* AsyncNetBase::firstTaskOp(int task_id) {
  return operator_nodes_[firstTaskOpId(task_id)].operator_.get();
}

OperatorBase* AsyncNetBase::lastTaskOp(int task_id) {
  return operator_nodes_[lastTaskOpId(task_id)].operator_.get();
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
}

void AsyncNetBase::handleChainError(
    int task_id,
    OperatorBase* op,
    const char* err_str,
    bool save_exception) noexcept {
  std::string err_msg = err_str;
  if (op) {
    err_msg += ",  op " + (op->has_debug_def() ? op->type() : " unknown");
  }
  LOG(ERROR) << err_msg;
  // mark end of chain with an error
  if (query(task_id) == EventStatus::EVENT_INITIALIZED) {
    if (save_exception) {
      event(task_id).SetFinishedWithException(err_msg.c_str());
    } else {
      event(task_id).SetFinished(err_msg.c_str());
    }
  }
}

bool AsyncNetBase::run(int task_id, int stream_id) noexcept {
  OperatorBase* op = nullptr;
  try {
    // Optionally insert async wait ops,
    // skip when finish_chain_ is set -
    // all parents are guaranteed to be finished
    if (!options_.finish_chain_) {
      asyncWait(task_id, stream_id, parents(task_id));
    }
    int iter_id = -1;
    if (tracer_) {
      iter_id = tracer_->getIter();
    }
    for (auto& op_id : chains_[task_id]) {
      op = operators_[op_id];
      bool success = false;
      if (!options_.report_stats_) {
        TRACE_EVENT(
            tracing::TRACE_OP,
            op_id,
            tracing::TRACE_TASK,
            task_id,
            tracing::TRACE_STREAM,
            stream_id,
            tracing::TRACE_ITER,
            iter_id);
        success = op->RunAsync(stream_id);
      } else {
        counters_.AddPerOpStartTime(op_id);
        success = op->RunAsync(stream_id);
        if (success && op->device_option().device_type() != PROTO_CPU) {
          op->Finish();
        }
        counters_.AddPerOpEndTime(op_id);
      }

      if (!success) {
        handleChainError(task_id, op, "Failed to execute an op");
        return false;
      }
    }

    op = nullptr;
    if (options_.finish_chain_) {
      operators_[chains_[task_id].back()]->event().Finish();
    }
  } catch (const std::exception& e) {
    handleChainError(task_id, op, e.what(), /* save_exception */ true);
    return false;
  } catch (...) {
    handleChainError(
        task_id,
        op,
        "Failed to execute task: unknown error",
        /* save_exception */ true);
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
  std::vector<OperatorBase*> pending_ops;
  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto status = query(task_id);
    if (status == EventStatus::EVENT_SCHEDULED) {
      // async cpu ops need to be handled separately,
      // as they may potentially never finish
      auto* op = lastTaskOp(task_id);
      if (op->HasAsyncPart() &&
          op->device_option().device_type() == PROTO_CPU) {
        pending_ops.push_back(op);
      } else {
        event(task_id).Finish();
      }
    } else if (status == EventStatus::EVENT_INITIALIZED) {
      event(task_id).SetFinished();
    }
  }

  // avoid events cancelling each other and causing
  // a deadlock
  std::atomic_flag error_happened = ATOMIC_FLAG_INIT;
  for (auto* pending_op : pending_ops) {
    pending_op->event().SetCallback(
        [pending_op, &pending_ops, &error_happened]() {
          // if one of the async cpu ops failed,
          // we have to terminate other pending async cpu ops
          auto status = pending_op->event().Query();
          TORCH_CHECK(
              status == EventStatus::EVENT_SUCCESS ||
              status == EventStatus::EVENT_FAILED);
          if (status == EventStatus::EVENT_FAILED) {
            // go through all the ops and terminate them,
            // we may get an exception in case of multiple
            // SetFinished() calls
            if (!error_happened.test_and_set()) {
              for (auto* op : pending_ops) {
                if (op != pending_op) {
                  try {
                    op->CancelAsyncCallback();
                    op->event().SetFinished("Cancelled");
                  } catch (const EnforceNotMet&) {
                    // ignore
                  }
                }
              }
            }
          }
        });
  }

  // wait for all pending ops to be finished or be terminated
  for (auto* pending_op : pending_ops) {
    pending_op->event().Finish();
  }

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    if (event(task_id).Query() != EventStatus::EVENT_SUCCESS) {
      success_ = false;
      break;
    }
  }
}

ProfDAGProtos AsyncNetBase::GetOperatorStats() const {
  return counters_.GetReport().GetOperatorStats();
}

ProfDAGProtos AsyncNetBase::GetPerOperatorCost() const {
  return counters_.GetReport().GetPerOperatorCost();
}

ProfDAGReport AsyncNetBase::GetProfReport() const {
  return counters_.GetReport();
}

AsyncNetBase::~AsyncNetBase() {
  if (options_.report_stats_) {
    counters_.GetReport().PrintStats();
  }
}

ExecutionOptions::ExecutionOptions(
    const std::shared_ptr<const NetDef>& net_def) {
  static const std::string kDag = "dag";
  static const std::string kProfDag = "prof_dag";
  static const std::string kAsyncDag = "async_dag";
  static const std::string kSimpleNet = "simple";

  std::string net_type;
  if (net_def->has_type() && !net_def->type().empty()) {
    net_type = net_def->type();
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
    report_stats_ = (net_type == kProfDag);
  } else if (net_type == kAsyncDag) {
    streams_per_gpu_ = 1;
    finish_chain_ = false;
    always_schedule_child_ = true;
    check_stream_status_ = false;
    use_single_pool_ = true;
    use_per_net_pools_ = true;
    is_blocking_ = true;
    report_stats_ = false;
  } else {
    streams_per_gpu_ = FLAGS_caffe2_streams_per_gpu;
    finish_chain_ = false;
    always_schedule_child_ = false;
    check_stream_status_ = FLAGS_caffe2_net_async_check_stream_status;
    use_single_pool_ = FLAGS_caffe2_net_async_use_single_pool;
    use_per_net_pools_ = FLAGS_caffe2_net_async_use_per_net_pools;
    is_blocking_ = false;
    report_stats_ = false;
  }

  use_dfs_scheduling_ = false;

  for (int arg_idx = 0; arg_idx < net_def->arg_size(); ++arg_idx) {
    auto& arg = net_def->arg(arg_idx);
    if (arg.has_name() && arg.name() == "enable_profiling") {
      CAFFE_ENFORCE(arg.has_i(), "enable_profiling should be an int");
      report_stats_ = arg.i() == 1;
    }
    if (arg.has_name() && arg.name() == "deferrable_mode") {
      CAFFE_ENFORCE(arg.has_i(), "deferrable_mode should be an int");
      use_dfs_scheduling_ = arg.i() == 1; // corr. to DFS scheduling
    }
  }

  if (FLAGS_caffe2_net_async_profile_operators) {
    report_stats_ = true;
  }
  run_root_tasks_inline_ = FLAGS_caffe2_net_async_run_root_tasks_inline;
}

} // namespace caffe2

namespace c10 {

C10_REGISTER_CREATOR(
    ThreadPoolRegistry,
    CPU,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CPU>);
C10_REGISTER_CREATOR(
    ThreadPoolRegistry,
    CUDA,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_CUDA>);
C10_REGISTER_CREATOR(
    ThreadPoolRegistry,
    HIP,
    caffe2::GetAsyncNetThreadPool<TaskThreadPool, caffe2::PROTO_HIP>);

} // namespace c10
