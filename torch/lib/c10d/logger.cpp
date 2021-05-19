#include <c10d/Utils.hpp>
#include <c10d/logger.hpp>
#include <fmt/format.h>
#include <string>

namespace c10d {

// When training runs at these iterations, log the runtime
// stats.
const int LoggingIterations[] = {10, 20, 100, 1000};

namespace {

const int kMilliSecondToNanosSecond = 1000000;

} // anonymous namespace

std::ostream& operator<<(std::ostream& output, const Logger& logger) {
  auto& ddp_logging_data = (*logger.ddp_logging_data_);

  std::string loggerInfo = fmt::format(
      "[Rank {} / {}] Training {} unused_parameter_size={} \n "
      "Avg forward compute time: {} \n Avg backward compute time: {} \n"
      "Avg backward comm. time: {} \n Avg backward comm/comp overlap time: {}",
      ddp_logging_data.ints_map["rank"],
      ddp_logging_data.ints_map["world_size"],
      ddp_logging_data.strs_map["module_name"],
      ddp_logging_data.ints_map["unused_parameter_size"],
      ddp_logging_data.ints_map["avg_forward_compute_time"],
      ddp_logging_data.ints_map["avg_backward_compute_time"],
      ddp_logging_data.ints_map["avg_backward_comm_time"],
      ddp_logging_data.ints_map["avg_backward_compute_comm_overlap_time"]);

  if (ddp_logging_data.strs_map["comm_hook"] != "") {
    loggerInfo += fmt::format(
        "\n Gradient comm. hook: {}", ddp_logging_data.strs_map["comm_hook"]);
  }

  if (ddp_logging_data.ints_map["join_uneven_inputs"]) {
    loggerInfo += "\n Uneven input detection with join() enabled.";
  }

  return output << loggerInfo;
}

Logger::Logger(std::shared_ptr<c10d::Reducer> reducer) {
  reducer_ = reducer;
  ddp_logging_data_ = std::make_unique<at::DDPLoggingData>();
}

// Environment variables
void Logger::set_env_variables() {
  ddp_logging_data_->strs_map["master_port"] = parse_env("MASTER_PORT");
  ddp_logging_data_->strs_map["master_addr"] = parse_env("MASTER_ADDR");
  ddp_logging_data_->strs_map["cuda_visible_devices"] =
      parse_env("CUDA_VISIBLE_DEVICES");
  if (reducer_->process_group_->getBackendName() == "nccl") {
    ddp_logging_data_->strs_map["nccl_socket_ifname"] =
        parse_env("NCCL_SOCKET_IFNAME");
    ddp_logging_data_->strs_map["nccl_blocking_wait"] =
        parse_env("NCCL_BLOCKING_WAIT");
    ddp_logging_data_->strs_map["nccl_async_error_handling"] =
        parse_env("NCCL_ASYNC_ERROR_HANDLING");
    ddp_logging_data_->strs_map["nccl_debug"] = parse_env("NCCL_DEBUG");
    ddp_logging_data_->strs_map["nccl_nthreads"] = parse_env("NCCL_NTHREADS");
    ddp_logging_data_->strs_map["nccl_ib_timeout"] =
        parse_env("NCCL_IB_TIMEOUT");
  }
  if (reducer_->process_group_->getBackendName() == "gloo") {
    ddp_logging_data_->strs_map["gloo_socket_ifname"] =
        parse_env("GLOO_SOCKET_IFNAME");
    ddp_logging_data_->strs_map["gloo_device_transport"] =
        parse_env("GLOO_DEVICE_TRANSPORT");
  }
}

void Logger::set_parameter_stats() {
  // The number of parameter tensors
  ddp_logging_data_->ints_map["num_parameter_tensors"] =
      reducer_->replicas_[0].size();
  // Total parameters size (Bytes)
  ddp_logging_data_->ints_map["total_parameter_size_bytes"] = 0;
  // Parameters' data types, there may be multiple data
  // types for mixed precision training.
  std::set<std::string> unique_dtypes;
  for (auto t : reducer_->replicas_[0]) {
    ddp_logging_data_->ints_map["total_parameter_size_bytes"] +=
        t.numel() * t.element_size();
    unique_dtypes.insert(std::string(t.dtype().name()));
  }
  ddp_logging_data_->strs_map["dtypes"] = c10::Join(", ", unique_dtypes);
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

// Communication hook. Empty string if not set, in which case it will not be
// logged.
void Logger::set_comm_hook(const std::string& hook) {
  ddp_logging_data_->strs_map["comm_hook"] = hook;
}

// Whether we are running under model.join() context manager for DDP uneven
// inputs.
void Logger::set_uneven_input_join() {
  ddp_logging_data_->ints_map["join_uneven_inputs"] = true;
}

void Logger::set_static_graph() {
  ddp_logging_data_->ints_map["static_graph"] = reducer_->static_graph_;
}

// Data that can be got during DistributedDataParallel construction time
void Logger::set_construction_data_and_log(
    const std::string& module_name,
    const std::vector<int>& device_ids,
    int output_device,
    bool broadcast_buffers) {
  // No lock is needed, as it will be called in DistributedDataParallel
  // constructor.
  ddp_logging_data_->strs_map["module_name"] = module_name;
  ddp_logging_data_->ints_map["world_size"] =
      reducer_->process_group_->getSize();
  ddp_logging_data_->ints_map["rank"] = reducer_->process_group_->getRank();
  // In which iteration of the training loop the get_ddp_logging_data()
  // is called to fetch the DDPLoggingData, 0 if the data is fetched
  // before training loop.
  ddp_logging_data_->ints_map["iteration"] = 0;
  ddp_logging_data_->ints_map["is_multi_device_module"] =
      reducer_->is_multi_device_module_;

  set_parameter_stats();
  // A list of bucket sizes (Bytes) calculated during construction time
  ddp_logging_data_->strs_map["bucket_sizes"] =
      c10::Join(", ", get_bucket_sizes());
  set_env_variables();

  // DistributedDataParallel constructor input parameters
  ddp_logging_data_->strs_map["device_ids"] = c10::Join(", ", device_ids);
  ddp_logging_data_->ints_map["output_device"] = output_device;
  ddp_logging_data_->ints_map["broadcast_buffers"] = broadcast_buffers;
  ddp_logging_data_->ints_map["bucket_cap_bytes"] = reducer_->bucket_bytes_cap_;
  ddp_logging_data_->ints_map["find_unused_parameters"] =
      reducer_->find_unused_parameters_;
  ddp_logging_data_->ints_map["gradient_as_bucket_view"] =
      reducer_->gradient_as_bucket_view_;
  ddp_logging_data_->strs_map["backend_name"] =
      reducer_->process_group_->getBackendName();

  if (parseDistDebugLevel() != DistributedDebugLevel::OFF) {
    std::string initInfo = fmt::format(
        "[Rank {}]: DDP Initialized with: \n",
        ddp_logging_data_->ints_map["rank"]);
    std::stringstream ddpLoggingDataInfo;
    for (const auto& intItem : ddp_logging_data_->ints_map) {
      ddpLoggingDataInfo << intItem.first << ": " << intItem.second << "\n";
    }
    for (const auto& strItem : ddp_logging_data_->strs_map) {
      ddpLoggingDataInfo << strItem.first << ": " << strItem.second << "\n";
    }
    LOG(INFO) << initInfo << ddpLoggingDataInfo.str();
  }

  at::LogPyTorchDDPUsage(*ddp_logging_data_);
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

void Logger::reset_performance_stats() {
  ddp_logging_data_->ints_map["forward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"] = 0;
}

void Logger::set_runtime_stats_and_log() {
  // Sync with reducer's data
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  // Set runtime stats at the sampling iterations.
  if (!reducer_->should_collect_runtime_stats()) {
    return;
  }
  num_iterations_stats_recorded_++;
  // Set ith iteration when the runtime stats are set.
  ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
  // When get_ddp_logging_data() is called, "unused_parameter_size",
  // "has_rebuilt_buckets" and "rebuilt_bucket_sizes" are updated in the latest
  // sampling iteration.
  // If unused_parameters_ is not empty, calculate its sizes.
  // unused_parameters_ is calculated in forward call of
  // each iteration.
  for (const auto& unused_index : reducer_->unused_parameters_) {
    const auto& v = reducer_->replicas_[unused_index.replica_index]
                                       [unused_index.variable_index];
    ddp_logging_data_->ints_map["unused_parameter_size"] +=
        v.numel() * v.element_size();
  }
  // rebuilt_bucket_sizes will not change once buckets are rebuilt,
  // so it only needs to set once during whole training loop.
  // Rebuild buckets stats after 1st iteration
  if (ddp_logging_data_->ints_map["has_rebuilt_buckets"] !=
      reducer_->has_rebuilt_bucket_) {
    ddp_logging_data_->ints_map["has_rebuilt_buckets"] =
        reducer_->has_rebuilt_bucket_;
    ddp_logging_data_->strs_map["rebuilt_bucket_sizes"] =
        c10::Join(", ", get_bucket_sizes());
  }

  reset_performance_stats();

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
        ddp_logging_data_->ints_map["avg_forward_compute_time"],
        ddp_logging_data_->ints_map["forward_compute_time"],
        reducer_->gpu_timer_.forward_start,
        reducer_->gpu_timer_.backward_compute_start);
    calculate_avg_gpu_time(
        ddp_logging_data_->ints_map["avg_backward_compute_time"],
        ddp_logging_data_->ints_map["backward_compute_time"],
        reducer_->gpu_timer_.backward_compute_start,
        reducer_->gpu_timer_.backward_compute_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->ints_map["avg_backward_comm_time"],
        ddp_logging_data_->ints_map["backward_comm_time"],
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_comm_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->ints_map["avg_backward_compute_comm_overlap_time"],
        ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"],
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_compute_end);
#endif
  } else {
    calculate_avg_cpu_time(
        ddp_logging_data_->ints_map["avg_forward_compute_time"],
        ddp_logging_data_->ints_map["forward_compute_time"],
        reducer_->cpu_timer_.forward_start_time,
        reducer_->cpu_timer_.backward_compute_start_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->ints_map["avg_backward_compute_time"],
        ddp_logging_data_->ints_map["backward_compute_time"],
        reducer_->cpu_timer_.backward_compute_start_time,
        reducer_->cpu_timer_.backward_compute_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->ints_map["avg_backward_comm_time"],
        ddp_logging_data_->ints_map["backward_comm_time"],
        reducer_->cpu_timer_.backward_comm_start_time,
        reducer_->cpu_timer_.backward_comm_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->ints_map["avg_backward_compute_comm_overlap_time"],
        ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"],
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
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  }
}

at::DDPLoggingData Logger::get_ddp_logging_data() {
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  return *ddp_logging_data_;
}

} // namespace c10d
