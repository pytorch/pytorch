#include <c10d/reducer.hpp>

namespace c10d {

class Logger {
 public:
  explicit Logger(std::shared_ptr<c10d::Reducer> reducer);
  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_logging_data(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers);

  // An Interface for users to get DDPLoggingData and log them
  // in the applications.
  c10::DDPLoggingData get_ddp_logging_data();

  // Set environment variables.
  void set_env_variables();
  // Set parameters stats.
  void set_parameter_stats();
  // Get size of each bucket (Bytes).
  std::vector<int> get_bucket_sizes();

  // Calculate avg stats using cpu timer and gpu timer
  // that has been recorded in reducer.
  void calculate_avg_cpu_time(
      int64_t& avg_time,
      int64_t cpu_start_time,
      int64_t cpu_end_time);
#ifdef USE_CUDA
  void calculate_avg_gpu_time(
      int64_t& avg_time,
      cudaEvent_t gpu_start,
      cudaEvent_t gpu_end);
#endif
  // Set stats that can be collected only during
  // training loop. It is called at beginning of forward call.
  void set_runtime_stats_and_log();

 private:
  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<c10::DDPLoggingData> ddp_logging_data_;
  std::shared_ptr<c10d::Reducer> reducer_;
};

} // namespace c10d
