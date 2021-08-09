#include <c10/util/Logging.h>
#include <c10d/reducer.hpp>

namespace c10d {

class TORCH_API Logger {
 public:
  explicit Logger(std::shared_ptr<c10d::Reducer> reducer);
  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_data_and_log(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers);

  void set_static_graph();

  // An interface for users to get DDPLoggingData and log them
  // in the applications. Explanation of logging fields are in
  // "struct DDPLoggingData" of "torch/c10/util/Logging.h".
  at::DDPLoggingData get_ddp_logging_data();

  // Stream insertion operator for logging data to stream under
  // TORCH_DISTRIBUTED_DEBUG.
  friend std::ostream& operator<<(std::ostream& output, const Logger& logger);

  ~Logger() noexcept(false) {
    // Log if DDP graph is static in Logger dtor instead of Reducer dtor since
    // Logger is deleted before Reducer.
    log_if_graph_static(reducer_->ddp_graph_static());
  }

  // Set environment variables.
  void set_env_variables();
  // Set parameters stats.
  void set_parameter_stats();
  // Get size of each bucket (Bytes).
  std::vector<int> get_bucket_sizes();
  // Get bucket size limits specified during DDP construction.
  std::vector<int> get_bucket_size_limits();
  // Get variable indices for each bucket.
  std::vector<std::vector<size_t>> get_per_bucket_variable_indices();
  // Set comm. hook, if used
  void set_comm_hook(const std::string& hook);
  // Set running with uneven input detection (model.join() context manager)
  void set_uneven_input_join();

  // Reset performance stats at current iteration
  void reset_performance_stats();

  // Calculate avg stats using cpu timer and gpu timer
  // that has been recorded in reducer.
  void calculate_avg_time(
      int64_t& avg_time,
      int64_t& time_duration,
      Timer& timer,
      Timer::Event start_event,
      Timer::Event end_event);

  // Set the absolute time of the event that has been recorded in reducer.
  void set_event_time(
    int64_t& event_time,
    Timer& timer,
    Timer::Event event
  );
  // Set stats that can be collected only during
  // training loop. It is called at the beginning of forward call
  // to record the run time stats of sampled iterations that previouly ran.
  // GPU performance stats are collected only for single process
  // single device program and single device module right now.
  // TODO to support single process multiple devices and multi device modules,
  // events need to be created and recorded on multiple devices.
  void set_runtime_stats_and_log();

  // Called when DDP/reducer is failing with an error. The
  // logging data structure will have two fields filled: "has_error" indicating
  // that this iteration encountered an error and other fields are not valid,
  // and "error", a string which contains the error message that DDP failed
  // with.
  template <typename... Args>
  void set_error_and_log(const std::string& ddp_error, const Args&... args) {
    ddp_logging_data_->ints_map["has_error"] = 1;
    auto err = c10::str(ddp_error, args...);
    ddp_logging_data_->strs_map["error"] = err;
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  }

  // When running without static graph, called when reducer is destroyed to log
  // if graph was actually static and is a candidate for static graph
  // optimization.
  void log_if_graph_static(bool is_static) {
    ddp_logging_data_->ints_map["can_set_static_graph"] = is_static;
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  }


 private:
  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<at::DDPLoggingData> ddp_logging_data_;
  std::shared_ptr<c10d::Reducer> reducer_;
  // track the number of iterations when runtime stats are collected so far.
  long num_iterations_stats_recorded_ = 0;
};

} // namespace c10d
