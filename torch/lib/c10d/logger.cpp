#include <c10d/logger.hpp>

namespace c10d {

namespace {

std::string parse_env(const char* env_var_name) {
  char* stringValue = std::getenv(env_var_name);
  std::string res = "N/A";
  if (stringValue != nullptr) {
    res = stringValue;
  }
  return res;
}

} // anonymous namespace

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
  ddp_logging_data_->nccl_debug = parse_env("NCCL_DEBUG");
  ddp_logging_data_->nccl_nthreads = parse_env("NCCL_NTHREADS");
  ddp_logging_data_->nccl_ib_timeout = parse_env("NCCL_IB_TIMEOUT");
}

void Logger::set_parameter_stats() {
  ddp_logging_data_->num_parameter_tensors = reducer_->replicas_[0].size();
  ddp_logging_data_->total_parameter_size_bytes = 0;
  for (const auto& t : reducer_->replicas_[0]) {
    ddp_logging_data_->total_parameter_size_bytes +=
        t.numel() * t.element_size();
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

void Logger::set_construction_logging_data(
    const std::string& module_name,
    const std::vector<int>& device_ids,
    int output_device,
    bool broadcast_buffers) {
  // Data that can be got during DistributedDataParallel construction time
  ddp_logging_data_->module_name = module_name;
  ddp_logging_data_->world_size = reducer_->process_group_->getSize();
  ddp_logging_data_->rank = reducer_->process_group_->getRank();
  ddp_logging_data_->iteration = 0;
  ddp_logging_data_->dtype =
      std::string(reducer_->replicas_[0][0].dtype().name());

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

  LogPyTorchDDPUsage(*ddp_logging_data_);
}

void Logger::calculate_avg_cpu_time(
    int64_t& avg_time,
    int64_t cpu_start_time,
    int64_t cpu_end_time) {
  long num_iters = reducer_->num_iterations_;
  avg_time = ((cpu_end_time - cpu_start_time) + avg_time * (num_iters - 1)) /
      num_iters;
}

#ifdef USE_CUDA
void Logger::calculate_avg_gpu_time(
    int64_t& avg_time,
    cudaEvent_t gpu_start,
    cudaEvent_t gpu_end) {
  long num_iters = reducer_->num_iterations_;
  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, gpu_start, gpu_end);
  avg_time =
      (int(milliseconds * 1000000) + avg_time * (num_iters - 1)) / num_iters;
}
#endif

void Logger::set_runtime_stats() {
  // set runtime stats after 1st iteration is complete.
  if (reducer_->num_iterations_ == 0) {
    return;
  }
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
  if (reducer_->num_iterations_ == 2) {
    ddp_logging_data_->has_rebuilt_buckets = reducer_->has_rebuilt_bucket_;
    ddp_logging_data_->rebuilt_bucket_sizes = get_bucket_sizes();
  }

  // set_runtime_stats is called at beginning of forward call,
  // when it is cheap to synchronize the cuda events of previous iteration,
  // as it is guaranteed that all cuda operations are done in previous
  // iteration.
  if (reducer_->replicas_[0][0].is_cuda()) {
#ifdef USE_CUDA
    cudaEventSynchronize(reducer_->gpu_timer_.forward_start);
    cudaEventSynchronize(reducer_->gpu_timer_.backward_compute_start);
    cudaEventSynchronize(reducer_->gpu_timer_.backward_compute_end);
    cudaEventSynchronize(reducer_->gpu_timer_.backward_comm_start);
    cudaEventSynchronize(reducer_->gpu_timer_.backward_comm_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_forward_compute_time,
        reducer_->gpu_timer_.forward_start,
        reducer_->gpu_timer_.backward_compute_start);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_compute_time,
        reducer_->gpu_timer_.backward_compute_start,
        reducer_->gpu_timer_.backward_compute_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_comm_time,
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_comm_end);
    calculate_avg_gpu_time(
        ddp_logging_data_->avg_backward_compute_comm_overlap_time,
        reducer_->gpu_timer_.backward_comm_start,
        reducer_->gpu_timer_.backward_compute_end);
#endif
  } else {
    calculate_avg_cpu_time(
        ddp_logging_data_->avg_forward_compute_time,
        reducer_->cpu_timer_.forward_start_time,
        reducer_->cpu_timer_.backward_compute_start_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_compute_time,
        reducer_->cpu_timer_.backward_compute_start_time,
        reducer_->cpu_timer_.backward_compute_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_comm_time,
        reducer_->cpu_timer_.backward_comm_start_time,
        reducer_->cpu_timer_.backward_comm_end_time);

    calculate_avg_cpu_time(
        ddp_logging_data_->avg_backward_compute_comm_overlap_time,
        reducer_->cpu_timer_.backward_comm_start_time,
        reducer_->cpu_timer_.backward_compute_end_time);
  }
}

c10::DDPLoggingData Logger::get_ddp_logging_data() {
  return *ddp_logging_data_;
}

} // namespace c10d
