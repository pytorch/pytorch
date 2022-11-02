#include <c10/util/StringUtil.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <string>

#include <c10/util/CallOnce.h>

#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif

namespace c10d {

// Logs runtime stats to configured destination. Note that since data collection
// only runs every ddp_runtime_logging_sample_rate iterations, the actual
// training iterations recorded will be like 10,
// (20-10) * ddp_runtime_logging_sample_rate,
// (50-10) * ddp_runtime_logging_sample_rate and so on.
const int LoggingIterations[] = {10, 20, 50, 100, 500, 800, 1000}; // NOLINT

std::ostream& operator<<(std::ostream& output, const Logger& logger) {
  auto& ddp_logging_data = (*logger.ddp_logging_data_);

  std::string loggerInfo = fmt::format(
      "[Rank {} / {}] [before iteration {}] Training {} unused_parameter_size={} \n "
      "Avg forward compute time: {} \n Avg backward compute time: {} \n"
      "Avg backward comm. time: {} \n Avg backward comm/comp overlap time: {}",
      ddp_logging_data.ints_map["rank"],
      ddp_logging_data.ints_map["world_size"],
      ddp_logging_data.ints_map["iteration"],
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

c10::once_flag log_graph_static_flag;

void Logger::log_if_graph_static(bool is_static) {
  c10::call_once(log_graph_static_flag, [this, is_static]() {
    ddp_logging_data_->ints_map["can_set_static_graph"] = is_static;
    // It is useful to report the iteration that training finished at.
    ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  });
}

// Environment variables
void Logger::set_env_variables() {
  ddp_logging_data_->strs_map["master_port"] = parse_env("MASTER_PORT");
  ddp_logging_data_->strs_map["master_addr"] = parse_env("MASTER_ADDR");
  ddp_logging_data_->strs_map["torch_distributed_debug"] =
      parse_env("TORCH_DISTRIBUTED_DEBUG");
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

#ifdef USE_C10D_GLOO
    // auto gloo_pg =
    //     static_cast<c10d::ProcessGroupGloo*>(reducer_->process_group_.get());
    // auto n_threads = gloo_pg->getNumThreads();
    // ddp_logging_data_->ints_map["gloo_num_threads"] = n_threads;
#endif
  }
}

void Logger::set_parameter_stats() {
  // The number of parameter tensors
  ddp_logging_data_->ints_map["num_parameter_tensors"] =
      reducer_->params_.size();
  // Total parameters size (Bytes)
  ddp_logging_data_->ints_map["total_parameter_size_bytes"] = 0;
  // Parameters' data types, there may be multiple data
  // types for mixed precision training.
  std::set<std::string> unique_dtypes;
  for (const auto& t : reducer_->params_) {
    ddp_logging_data_->ints_map["total_parameter_size_bytes"] +=
        t.numel() * t.element_size();
    unique_dtypes.insert(std::string(t.dtype().name()));
  }
  ddp_logging_data_->strs_map["dtypes"] = c10::Join(", ", unique_dtypes);
}

std::vector<std::vector<size_t>> Logger::get_per_bucket_variable_indices() {
  std::vector<std::vector<size_t>> per_bucket_variable_indices;
  per_bucket_variable_indices.reserve(reducer_->buckets_.size());
  for (const auto& bucket : reducer_->buckets_) {
    const auto& indices = bucket.variable_indices;
    per_bucket_variable_indices.push_back(indices);
  }
  return per_bucket_variable_indices;
}

std::vector<int64_t> Logger::get_bucket_sizes() {
  std::vector<int64_t> bucket_sizes;
  for (const auto& bucket : reducer_->buckets_) {
    const auto& variables = bucket.variables;
    int64_t bucket_size = 0;
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
    bool broadcast_buffers,
    bool has_sync_bn,
    bool static_graph) {
  // No lock is needed, as it will be called in DistributedDataParallel
  // constructor.
  if (static_graph) {
    set_static_graph();
  }
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
  ddp_logging_data_->ints_map["has_sync_bn"] = has_sync_bn;
  ddp_logging_data_->ints_map["bucket_cap_bytes"] = reducer_->bucket_bytes_cap_;
  ddp_logging_data_->ints_map["find_unused_parameters"] =
      reducer_->find_unused_parameters_;
  ddp_logging_data_->ints_map["gradient_as_bucket_view"] =
      reducer_->gradient_as_bucket_view_;
  ddp_logging_data_->strs_map["backend_name"] =
      reducer_->process_group_->getBackendName();

  if (debug_level() != DebugLevel::Off) {
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

void Logger::set_event_time(
    int64_t& event_time,
    Timer& timer,
    Timer::Event event) {
  auto timestamp = timer.getTimestamp(event);
  if (timestamp != c10::nullopt) {
    // TODO: should we set this as human-readable time instead of unixtime?
    event_time = *timestamp;
  }
}

void Logger::calculate_avg_time(
    int64_t& avg_time,
    int64_t& time_duration,
    Timer& timer,
    Timer::Event start_event,
    Timer::Event end_event) {
  TORCH_CHECK(num_iterations_stats_recorded_ > 0);
  c10::optional<int64_t> maybe_time_duration =
      timer.measureDifference(start_event, end_event);
  if (!maybe_time_duration.has_value()) {
    return;
  }
  time_duration = maybe_time_duration.value();
  avg_time = (time_duration + avg_time * (num_iterations_stats_recorded_ - 1)) /
      num_iterations_stats_recorded_;
}

void Logger::reset_performance_stats() {
  ddp_logging_data_->ints_map["forward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"] = 0;
  ddp_logging_data_->ints_map["forward_compute_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time_end"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time_end"] = 0;
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
  if (reducer_->unused_parameters_.size() == 0 &&
      reducer_->find_unused_parameters_) {
    // No unused params in this iteration
    ddp_logging_data_->ints_map["unused_parameter_size"] = 0;
  }
  for (const auto& unused_index : reducer_->unused_parameters_) {
    const auto& v = reducer_->params_[unused_index];
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
    // Log per-bucket variable indices
    std::vector<std::string> per_bucket_variable_indices;
    auto indices = get_per_bucket_variable_indices();
    per_bucket_variable_indices.reserve(indices.size());
    for (const auto& bucket_indices : indices) {
      per_bucket_variable_indices.push_back(c10::Join(" ", bucket_indices));
    }
    ddp_logging_data_->strs_map["rebuilt_per_bucket_param_indices"] =
        c10::Join(", ", per_bucket_variable_indices);
  }
  // Log gradient ready order
  if (!reducer_->grad_ready_order_indices_.empty()) {
    // Note that the indices are for the previous iteration as
    // this function is called in forward pass, and we last computed gradient
    // ready order in the last backward pass.
    ddp_logging_data_->strs_map["prev_iteration_grad_ready_order_indices"] =
        c10::Join(", ", reducer_->grad_ready_order_indices_);
  }

  reset_performance_stats();

  // Cuda time stats are only collected for single device modules.
  if (reducer_->params_[0].is_cuda() && reducer_->is_multi_device_module_) {
    TORCH_WARN_ONCE(
        "Cuda time stats are not collected for multi-device modules.");
    return;
  }
  if (!reducer_->params_[0].is_cuda() && !reducer_->params_[0].is_cpu()) {
    TORCH_WARN_ONCE(
        "Time stats are currently only collected for CPU and CUDA devices. "
        "Please refer to CpuTimer or CudaTimer for how to register timer "
        "for other device type.");
    return;
  }
  TORCH_INTERNAL_ASSERT(reducer_->timer_);
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_forward_compute_time"],
      ddp_logging_data_->ints_map["forward_compute_time"],
      *reducer_->timer_,
      Timer::Event::kForwardStart,
      Timer::Event::kBackwardComputeStart);
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_compute_time"],
      ddp_logging_data_->ints_map["backward_compute_time"],
      *reducer_->timer_,
      Timer::Event::kBackwardComputeStart,
      Timer::Event::kBackwardComputeEnd);
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_comm_time"],
      ddp_logging_data_->ints_map["backward_comm_time"],
      *reducer_->timer_,
      Timer::Event::kBackwardCommStart,
      Timer::Event::kBackwardCommEnd);
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_compute_comm_overlap_time"],
      ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"],
      *reducer_->timer_,
      Timer::Event::kBackwardCommStart,
      Timer::Event::kBackwardComputeEnd);

  set_event_time(
      ddp_logging_data_->ints_map["forward_compute_time_start"],
      *reducer_->timer_,
      Timer::Event::kForwardStart);
  set_event_time(
      ddp_logging_data_->ints_map["backward_compute_time_start"],
      *reducer_->timer_,
      Timer::Event::kBackwardComputeStart);
  set_event_time(
      ddp_logging_data_->ints_map["backward_comm_time_start"],
      *reducer_->timer_,
      Timer::Event::kBackwardCommStart);
  set_event_time(
      ddp_logging_data_->ints_map["backward_compute_time_end"],
      *reducer_->timer_,
      Timer::Event::kBackwardComputeEnd);
  set_event_time(
      ddp_logging_data_->ints_map["backward_comm_time_end"],
      *reducer_->timer_,
      Timer::Event::kBackwardCommEnd);

  // Log runtime stats to stderr if TORCH_DISTRIBUTED_DEBUG=DETAIL is enabled.
  if (debug_level() == DebugLevel::Detail) {
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
