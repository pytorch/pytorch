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

void Logger::set_bucket_stats() {
  for (const auto& bucket : reducer_->buckets_) {
    const auto& variables = bucket.replicas[0].variables;
    int bucket_size = 0;
    for (const auto& v : variables) {
      bucket_size += v.numel() * v.element_size();
    }
    ddp_logging_data_->bucket_sizes.push_back(bucket_size);
  }
}

void Logger::set_construction_data_and_log(
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
  set_bucket_stats();
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

c10::DDPLoggingData Logger::get_ddp_logging_data() {
  return *ddp_logging_data_;
}

} // namespace c10d
