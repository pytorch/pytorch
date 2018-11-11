#include "caffe2/core/jit/net_jit_executor.h"

#include "caffe2/core/net_async_base.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

JITExecutor::JITExecutor(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws), run_future_(nullptr) {
  jit_ = std::make_shared<JITC2Program>(net_def, ws);
  num_workers_ = net_def->num_workers();
  CAFFE_ENFORCE_GT(
      num_workers_, 0, "Expected positive number of worker threads");

  checkNetArguments();

  LOG(INFO) << "Initialized JITExecutor: '" << Name()
            << "', #ops: " << net_def->op_size()
            << ", #workers: " << num_workers_
            << ", #jit ops: " << jit_->GetOps().size();
}

// RunAsync(), Wait() - not thread safe
bool JITExecutor::RunAsync() {
  CAFFE_ENFORCE(!run_future_ || run_future_->IsCompleted(), "Concurrent runs");
  reset();
  StartAllObservers();

  // Run an initial task starting from address 0 of a JIT program
  run_future_ =
      RunTask(std::make_shared<JITC2Task>(jit_, 0, *this, use_dfs_scheduling_));

  if (run_future_) {
    run_future_->SetCallback([this](const JITFuture* f) {
      StopAllObservers();
      finishRun();
    });
    return true;
  } else {
    LOG(ERROR) << "Failed to start an async run: " << net_def_->name();
    StopAllObservers();
    return false;
  }
}

bool JITExecutor::handleRunError() {
  CAFFE_ENFORCE(
      run_future_ && run_future_->IsCompleted(), "Expected a finished run");
  // TODO: exceptions from ops
  return !run_future_->IsFailed();
}

void JITExecutor::Wait() {
  CAFFE_ENFORCE(run_future_, "Expected a run in progress");
  run_future_->Wait();
}

JITFuture* JITExecutor::RunTask(const std::shared_ptr<JITC2Task>& task) {
  std::unique_lock<std::mutex> lock(tasks_mutex_);
  tasks_.push_back(task);
  auto* t = tasks_.back().get();
  Pool(t->GetDeviceOption())->run([t]() { t->Run(); });
  return &t->GetFuture();
}

void JITExecutor::reset() {
  run_future_ = nullptr;
  tasks_.clear();
}

void JITExecutor::checkNetArguments() {
  for (int arg_idx = 0; arg_idx < net_def_->arg_size(); ++arg_idx) {
    auto& arg = net_def_->arg(arg_idx);
    if (arg.has_name() && arg.name() == "deferrable_mode") {
      CAFFE_ENFORCE(arg.has_i(), "deferrable_mode should be an int");
      use_dfs_scheduling_ = arg.i() == 1; // corr. to DFS scheduling
      if (use_dfs_scheduling_) {
        LOG(INFO) << "Using DFS scheduling (" << Name() << ")";
      }
    }
  }
}

TaskThreadPoolBase* JITExecutor::poolGetter(
    PoolsMap& pools,
    int device_type,
    int device_id,
    int pool_size) {
  std::unique_lock<std::mutex> pools_lock(pools_mutex_);
  auto pool = pools[device_id][pool_size];
  if (!pool) {
    pool = ThreadPoolRegistry()->Create(
        DeviceTypeName(device_type),
        device_id,
        pool_size,
        /* create_new */ false);
    pools[device_id][pool_size] = pool;
  }
  return pool.get();
}

TaskThreadPoolBase* JITExecutor::Pool(const DeviceOption& device_option) {
  if (IsCPUDeviceType(device_option.device_type())) {
    auto numa_node_id = -1;
    if (device_option.has_device_id()) {
      numa_node_id = device_option.device_id();
      CAFFE_ENFORCE_GE(numa_node_id, 0, "Invalid NUMA node id: ", numa_node_id);
    }
    CAFFE_ENFORCE_LT(
        numa_node_id,
        FLAGS_caffe2_net_async_max_numa_nodes,
        "Invalid NUMA node id: ",
        numa_node_id);
    return poolGetter(cpu_pools_, PROTO_CPU, numa_node_id, num_workers_);
  } else if (device_option.device_type() == PROTO_CUDA) {
    auto gpu_id = device_option.device_id();
    CAFFE_ENFORCE(
        gpu_id >= 0 && gpu_id < FLAGS_caffe2_net_async_max_gpus,
        "Invalid GPU id: " + caffe2::to_string(gpu_id));
    return poolGetter(gpu_pools_, PROTO_CUDA, gpu_id, num_workers_);
  } else {
    CAFFE_THROW(
        "Unsupported device type " +
        caffe2::to_string(device_option.device_type()));
  }
}

JITExecutor::~JITExecutor() {}

REGISTER_NET(jit_executor, JITExecutor);

} // namespace caffe2
