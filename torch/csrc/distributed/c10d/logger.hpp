#pragma once

#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <utility>

namespace c10d {

// A struct to hold the latest status of the process group.
struct ProcessGroupStatus {
  // the sequential number of the last collective enqueued into workMetaList_
  // This is useful for identifying a rank that has not join a collective
  // initialized to be -1 to indicate no collective has been enqueued
  int64_t lastEnqueuedSeq{-1};
  // the sequential number of the last collective started as the kernel
  int64_t lastStartedSeq{-1};
  // the sequential number of the last collective completed marked by
  // the watchdog thread
  // initialized to be -1 to indicate no collective has been completed
  int64_t lastCompletedSeq{-1};

  // the name of the last collective enqueued into workMetaList_
  std::string lastEnqueuedWorkName;
  // the name of the last collective started as the kernel
  std::string lastStartedWorkName;
  // the name of the last collective completed
  std::string lastCompletedWorkName;

  // the sizes of the last work enqueued
  size_t lastEnqueuedNumelIn;
  size_t lastEnqueuedNumelOut;
  // the sizes of the last work completed
  size_t lastCompletedNumelIn;
  size_t lastCompletedNumelOut;
  // the sizes of the last work started
  size_t lastStartedNumelIn;
  size_t lastStartedNumelOut;
};

class TORCH_API Logger {
 public:
  explicit Logger(std::shared_ptr<c10d::Reducer> reducer);
  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_data_and_log(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers,
      bool has_sync_bn,
      bool static_graph);

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
  std::vector<int64_t> get_bucket_sizes();
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
  void set_event_time(int64_t& event_time, Timer& timer, Timer::Event event);
  // Set stats that can be collected only during
  // training loop. It is called at the beginning of forward call
  // to record the run time stats of sampled iterations that previously ran.
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
    // Report the iteration we are erroring at so user knows how many examples
    // successfully processed before this error was hit.
    ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  }

  // When running without static graph, called when reducer is destroyed to log
  // if graph was actually static and is a candidate for static graph
  // optimization.
  void log_if_graph_static(bool is_static);

 private:
  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<at::DDPLoggingData> ddp_logging_data_;
  std::shared_ptr<c10d::Reducer> reducer_;
  // track the number of iterations when runtime stats are collected so far.
  long num_iterations_stats_recorded_ = 0;
};

// a generic logging data struct that holds different types of logging data.
// starting with key value pairs of strings and integers,
// It can be extended to more types as needed.
struct C10dLoggingData {
  // logging fields that are string types.
  std::map<std::string, std::string> strings;
  // logging fields that are int64_t types.
  std::map<std::string, int64_t> integers;
};

class TORCH_API C10dLogger {
 public:
  C10dLogger(const C10dLogger&) = default;
  C10dLogger(C10dLogger&&) = delete;
  C10dLogger& operator=(const C10dLogger&) = default;
  C10dLogger& operator=(C10dLogger&&) = delete;
  virtual ~C10dLogger() = default;
  virtual void log(const C10dLoggingData& data);
  static C10dLogger* getLogger();
  static void registerLogger(std::unique_ptr<C10dLogger>);

 protected:
  // singletion, hide constructor from the public
  C10dLogger(std::string logDestination)
      : logDestination_(std::move(logDestination)) {}

  // the name of the destination this logger should log to
  std::string logDestination_;

 private:
  static std::unique_ptr<C10dLogger> logger_;
  static std::atomic<bool> registered_;
};

} // namespace c10d
