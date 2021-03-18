#include <c10d/Utils.hpp>
#include <c10d/logger.hpp>
#include <fmt/format.h>

namespace c10d {

// When training runs at these iterations, log the runtime
// stats.
const int LoggingIterations[] = {10, 20, 100, 1000};

namespace {

const int kMilliSecondToNanosSecond = 1000000;

} // anonymous namespace

std::ostream& operator<<(std::ostream& output, const Logger& logger) {
  auto& ddp_logging_data = logger.ddp_logging_data_;

  std::string loggerInfo = fmt::format(
      "[Rank {} / {}] Training {} unused_parameter_size={} \n "
      "Avg forward compute time: {} \n Avg backward compute time: {} \n"
      "Avg backward comm. time: {} \n Avg backward comm/comp overlap time: {}",
      ddp_logging_data->rank,
      ddp_logging_data->world_size,
      ddp_logging_data->module_name,
      ddp_logging_data->unused_parameter_size,
      ddp_logging_data->avg_forward_compute_time,
      ddp_logging_data->avg_backward_compute_time,
      ddp_logging_data->avg_backward_comm_time,
      ddp_logging_data->avg_backward_compute_comm_overlap_time);

  if (ddp_logging_data->comm_hook != "") {
    loggerInfo +=
        fmt::format("\n Gradient comm. hook: {}", ddp_logging_data->comm_hook);
  }

  return output << loggerInfo;
}

Logger::Logger(std::shared_ptr<c10d::Reducer> reducer) {
  reducer_ = reducer;
  ddp_logging_data_ = std::make_unique<c10::DDPLoggingData>();
}

void Logger::set_env_variables() {
  // Environment variables
  ddp_logging_data_->master_port = parse_env("MASTER_PORT");
  ddp_logging_data_->master_addr = parse_env("MASTER_ADDR");
  ddp_logging_data_->cuda_visible_devices = parse_env("CUDA_VISIBLE_DEVICES");
  ddp_logging_data_->gloo_socket_ifname = parse_env("GLOO_SOCKET_IFNAME");
  ddp_logging_data_->gloo_device_transport = parse_env("GLOO_DEVICE_TRANSPORT");
  ddp_logging_data_->nccl_socket_ifname = parse_env("NCCL_SOCKET_IFNAME");
  ddp_logging_data_->nccl_blocking_wait = parse_env("NCCL_BLOCKING_WAIT");
  ddp_logging_data_->nccl_async_error_handling =
      parse_env("NCCL_ASYNC_ERROR_HANDLING");
  ddp_logging_data_->nccl_debug = parse_env("NCCL_DEBUG");
  ddp_logging_data_->nccl_nthreads = parse_env("NCCL_NTHREADS");
  ddp_logging_data_->nccl_ib_timeout = parse_env("NCCL_IB_TIMEOUT");
}

void Logger::set_parameter_stats() {
  ddp_logging_data_->num_parameter_tensors = reducer_->replicas_[0].size();
  ddp_logging_data_->total_parameter_size_bytes = 0;
  std::set<std::string> unique_dtypes;
  for (auto t : reducer_->replicas_[0]) {
    ddp_logging_data_->total_parameter_size_bytes +=
        t.numel() * t.element_size();
    unique_dtypes.insert(std::string(t.dtype().name()));
  }
  for (auto dtype : unique_dtypes) {
    ddp_logging_data_->dtypes.push_back(dtype);
  }
}

std::vector<int> Logger::get_bucket_sizes() {
  std::vector<int> bucket_sizes;
  for (const auto& bucket : reducer_->buckets_) {
    const auto& variables = bucket.replicas[0].variables;
    int bucket_size = 0;
    for (const auto& v : variables) {
      bucket_size += v.numel() * v.element_size();
    }
    bucket_sizes.push_back(bucket_size);
  }
  return bucket_sizes;
}

void Logger::set_comm_hook(const std::string& hook) {
  ddp_logging_data_->comm_hook = hook;
}

void Logger::set_construction_data_and_log(
    const std::string& module_name,
    const std::vector<int>& device_ids,
    int output_device,
    bool broadcast_buffers) {
  // No lock is needed, as it will be called in DistributedDataParallel
  // constructor.
  // Data that can be got during DistributedDataParallel construction time.
  ddp_logging_data_->module_name = module_name;
  ddp_logging_data_->world_size = reducer_->process_group_->getSize();
  ddp_logging_data_->rank = reducer_->process_group_->getRank();
  ddp_logging_data_->iteration = 0;
  ddp_logging_data_->is_multi_device_module = reducer_->is_multi_device_module_;

  set_parameter_stats();
  ddp_logging_data_->bucket_sizes = get_bucket_sizes();
  set_env_variables();

  // DistributedDataParallel constructor input parameters
  ddp_logging_data_->device_ids = device_ids;
  ddp_logging_data_->output_device = output_device;
  ddp_logging_data_->broadcast_buffers = broadcast_buffers;
  ddp_logging_data_->bucket_cap_mb =
      (float)reducer_->bucket_bytes_cap_ / (1024 * 1024);
  ddp_logging_data_->find_unused_parameters = reducer_->find_unused_parameters_;
  ddp_logging_data_->gradient_as_bucket_view =
      reducer_->gradient_as_bucket_view_;
  ddp_logging_data_->backend_name = reducer_->process_group_->getBackendName();

  if (parseDistDebugLevel() != DistributedDebugLevel::OFF) {
    std::string initInfo = fmt::format(
        "[Rank {}]: DDP Initialized with: \n", ddp_logging_data_->rank);
    LOG(INFO) << initInfo << *ddp_logging_data_;
  }

  LogPyTorchDDPUsage(*ddp_logging_data_);
}

void Logger::calculate_avg_cpu_time(
    int64_t& avg_time,
    int64_t& time_duration,
    int64_t cpu_start_time,
    int64_t cpu_end_time) {
  // If cpu_end_time is not recorded in this iteration,
  // avg_time will return invalid value.
  // For some cases like DDP runs on non-sync mode, backward compute
  // end time can not be recorded in this iteration and thus can not
  // calculate the valid avg_time.
  // In this case, skip calculating the avg_time and return.
  TORCH_CHECK(num_iterations_stats_recorded_ > 0);
  if (cpu_end_time < cpu_start_time) {
    return;
  }
  time_duration = cpu_end_time - cpu_start_time;
  avg_time = (time_duration + avg_time * (num_iterations_stats_recorded_ - 1)) /
      num_iterations_stats_recorded_;
}

#ifdef USE_CUDA
void Logger::calculate_avg_gpu_time(
    int64_t& avg_time,
    int64_t& time_duration,
    at::cuda::CUDAEvent& gpu_start,
    at::cuda::CUDAEvent& gpu_end) {
  TORCH_CHECK(num_iterations_stats_recorded_ > 0);
  float milliseconds = gpu_start.elapsed_time(gpu_end);
  // If gpu_end is not recorded in this iteration,
  // milliseconds will have invalid value.
  // For some cases like DDP runs on non-sync mode,
  // gpu_end can not be recorded in this iteration and thus can not
  // calculate the valid avg_time.
  // In this case, skip calculating the avg_time and return.
  if (milliseconds < 0) {
    return;
  }
  time_duration = int64_t(milliseconds * kMilliSecondToNanosSecond);
  avg_time = (time_duration + avg_time * (num_iterations_stats_recorded_ - 1)) /
      num_iterations_stats_recorded_;
}
#endif

void Logger::set_runtime_stats_and_log() {
  // Sync with reducer's data
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  // Set runtime stats at the sampling iterations.
  if (!reducer_->should_collect_runtime_stats()) {
    return;
  }
  num_iterations_stats_recorded_++;
  // Set ith iteration when the runtime stats are set.
  ddp_logging_data_->iteration = reducer_->num_iterations_;
  // If unused_parameters_ is not empty, calculate its sizes.
  // unused_parameters_ is calculated in forward call of
  // each iteration.
  for (const auto& unused_index : reducer_->unused_parameters_) {
    const auto& v = reducer_->replicas_[unused_index.replica_index]
                                       [unused_index.variable_index];
    ddp_logging_data_->unused_parameter_size += v.numel() * v.element_size();
  }
  // rebuilt_bucket_sizes will not change once buckets are rebuilt,
  // so it only needs to set once during whole training loop.
  if (ddp_logging_data_->has_rebuilt_buckets != reducer_->has_rebuilt_bucket_) {
    ddp_logging_data_->has_rebuilt_buckets = reducer_->has_rebuilt_bucket_;
    ddp_logging_data_->rebuilt_bucket_sizes = get_bucket_sizes();
  }

  if (reducer_->replicas_[0][0].is_cuda()) {
#ifdef USE_CUDA
    // Cuda time stats are only collected for single process single
    // device and single device module.
    if (reducer_->replicas_.size() > 1 || reducer_->is_multi_device_module_) {
      TORCH_WARN_ONCE(
          "Cuda time stats are not collected for single process "
          "multiple device program or multi-device modules.");
      return;
    }
    // Check events on the replicas_[0][0].device().
    at::DeviceGuard g(reducer_->replicas_[0][0].device());
    // It is possible users did not call backward or run codes in
    // no-sync mode, in this case, some cudaEvents like "backward_compute_end"
    // or "backward_comm_start" or "backward_comm_end" will not be recorded.
    // cudaEvent is created when it is first time to be recorded.
    // If it is never recorded/created, skip synchronize and calculation.
    // Otherwise it will throw cuda errors.
    if (!reducer_->gpu_timer_.forward_start.isCreated() ||
        !reducer_->gpu_timer_.backward_compute_start.isCreated() ||
        !reducer_->gpu_timer_.backward_compute_end.isCreated() ||
        !reducer_->gpu_timer_.backward_comm_start.isCreated() ||
        !reducer_->gpu_timer_.backward_comm_end.isCreated()) {
      return;
    }

    // set_runtime_stats_and_log is called at the beginning of forward call,
    // when it is cheap to synchronize the cuda events of previous iteration,
    // as mostly all cuda operations are finished in previous iteration.
    reducer_->gpu_timer_.forward_start.synchronize();
    reducer_->gpu_timer_.backward_compute_start.synchronize();
    reducer_->gpu_timer_.backward_compute_end.synchronize();
    reducer_->gpu_timer_.backward_comm_start.synchronize();
    reducer_->gpu_timer_.backward_comm_end.synchronize();
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_forward_compute_time,
        ddp_logging_data_->forward_compute_time,
        reducer_->gpu_timer_.forward_start,
        reducer_->gpu_timer_.backward_compute_start);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_compute_time,
        ddp_logging_data_->backward_compute_time,
        reducer_->gpu_timer_.backward_compute_start,
        reducer_->gpu_timer_.backward_compute_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_comm_time,
        ddp_logging_data_->backward_comm_time,
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_comm_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_compute_comm_overlap_time,
        ddp_logging_data_->backward_compute_comm_overlap_time,
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_compute_end);
#endif
  } else {
    calculate_avg_cpu_time(
        ddp_logging_data_->avg_forward_compute_time,
        ddp_logging_data_->forward_compute_time,
        reducer_->cpu_timer_.forward_start_time,
        reducer_->cpu_timer_.backward_compute_start_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_compute_time,
        ddp_logging_data_->backward_compute_time,
        reducer_->cpu_timer_.backward_compute_start_time,
        reducer_->cpu_timer_.backward_compute_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_comm_time,
        ddp_logging_data_->backward_comm_time,
        reducer_->cpu_timer_.backward_comm_start_time,
        reducer_->cpu_timer_.backward_comm_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_compute_comm_overlap_time,
        ddp_logging_data_->backward_compute_comm_overlap_time,
        reducer_->cpu_timer_.backward_comm_start_time,
        reducer_->cpu_timer_.backward_compute_end_time);
  }
  // Log runtime stats to stderr if TORCH_DISTRIBUTED_DEBUG=DETAIL is enabled.
  if (parseDistDebugLevel() == DistributedDebugLevel::DETAIL) {
    LOG(INFO) << *this;
  }

  // Log runtime (e.g. avg performance) stats at the beginning and also
  // after a larger number of iterations. Choosing 10/1000/10000 is
  // not scientific here, it assumes most of applications will run
  // at least 10 iterations. stats could have smaller variance if
  // selected num_iterations_ is larger.
  if (std::find(
          std::begin(LoggingIterations),
          std::end(LoggingIterations),
          num_iterations_stats_recorded_) != std::end(LoggingIterations)) {
    LogPyTorchDDPUsage(*ddp_logging_data_);
  }
}

c10::DDPLoggingData Logger::get_ddp_logging_data() {
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  return *ddp_logging_data_;
}

} // namespace c10d
