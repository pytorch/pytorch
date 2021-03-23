#include <c10d/reducer.hpp>

namespace c10d {

class Logger {
 public:
  explicit Logger(std::shared_ptr<c10d::Reducer> reducer);
  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_data_and_log(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers);

  // An interface for users to get DDPLoggingData and log them
  // in the applications. Explanation of logging fields are in
  // "struct DDPLoggingData" of "torch/c10/util/Logging.h".
  c10::DDPLoggingData get_ddp_logging_data();

  // Stream insertion operator for logging data to stream under
  // TORCH_DISTRIBUTED_DEBUG.
  friend std::ostream& operator<<(std::ostream& output, const Logger& logger);

  // Set environment variables.
  void set_env_variables();
  // Set parameters stats.
  void set_parameter_stats();
  // Get size of each bucket (Bytes).
  std::vector<int> get_bucket_sizes();
  // Set comm. hook, if used
  void set_comm_hook(const std::string& hook);

  // Calculate avg stats using cpu timer and gpu timer
  // that has been recorded in reducer.
  void calculate_avg_cpu_time(
      int64_t& avg_time,
      int64_t& time_duration,
      int64_t cpu_start_time,
      int64_t cpu_end_time);
#ifdef USE_CUDA
  void calculate_avg_gpu_time(
      int64_t& avg_time,
      int64_t& time_duration,
      at::cuda::CUDAEvent& gpu_start,
      at::cuda::CUDAEvent& gpu_end);
#endif
  // Set stats that can be collected only during
  // training loop. It is called at the beginning of forward call
  // to record the run time stats of sampled iterations that previouly ran.
  // GPU performance stats are collected only for single process
  // single device program and single device module right now.
  // TODO to support single process multiple devices and multi device modules,
  // events need to be created and recorded on multiple devices.
  void set_runtime_stats_and_log();

 private:
  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<c10::DDPLoggingData> ddp_logging_data_;
  std::shared_ptr<c10d::Reducer> reducer_;
  // track the number of iterations when runtime stats are collected so far.
  long num_iterations_stats_recorded_ = 0;
};

} // namespace c10d
