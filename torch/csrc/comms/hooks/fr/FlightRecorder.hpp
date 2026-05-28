// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/FileSystem.h>
#include <torch/csrc/comms/RemovableHandle.hpp>
#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/utils/Utils.hpp>
#include <torch/csrc/profiler/combined_traceback.h>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace torch {
namespace comms {
namespace fr {

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

#define DEFINE_CONSTANT(name, value)     \
  static const c10::IValue name = value; \
  static const std::string name##_str = value;
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
// Also update both JSON and Pickle dumps to make use of the newly defined
// field(s).
DEFINE_CONSTANT(version_val, "2.10")
DEFINE_CONSTANT(entries_key, "entries")
DEFINE_CONSTANT(nccl_comm_key, "nccl_comm_state")
DEFINE_CONSTANT(comm_lib_version_key, "comm_lib_version")
DEFINE_CONSTANT(version_key, "version")
DEFINE_CONSTANT(pg_config_key, "pg_config")
DEFINE_CONSTANT(pg_status_key, "pg_status")
DEFINE_CONSTANT(record_id_key, "record_id")
DEFINE_CONSTANT(pg_id_key, "pg_id")
DEFINE_CONSTANT(pg_name_key, "process_group")
DEFINE_CONSTANT(collective_seq_id_key, "collective_seq_id")
// TODO: remove p2p information from the flight recorder output
// once it is handled in the analyzer
DEFINE_CONSTANT(p2p_seq_id_key, "p2p_seq_id")
DEFINE_CONSTANT(is_p2p_key, "is_p2p")
DEFINE_CONSTANT(op_id_key, "op_id")
DEFINE_CONSTANT(profiling_name_key, "profiling_name")
DEFINE_CONSTANT(input_sizes_key, "input_sizes")
DEFINE_CONSTANT(input_dtypes_key, "input_dtypes")
DEFINE_CONSTANT(output_sizes_key, "output_sizes")
DEFINE_CONSTANT(output_dtypes_key, "output_dtypes")
DEFINE_CONSTANT(time_created_key, "time_created_ns")
DEFINE_CONSTANT(duration_key, "duration_ms")
DEFINE_CONSTANT(timeout_key, "timeout_ms")
DEFINE_CONSTANT(frames_key, "frames")
DEFINE_CONSTANT(state_key, "state")
DEFINE_CONSTANT(line_key, "line")
DEFINE_CONSTANT(name_key, "name")
DEFINE_CONSTANT(filename_key, "filename")
DEFINE_CONSTANT(retired_key, "retired")
DEFINE_CONSTANT(time_discovered_started_key, "time_discovered_started_ns")
DEFINE_CONSTANT(time_discovered_completed_key, "time_discovered_completed_ns")
DEFINE_CONSTANT(completed_state, "completed")
DEFINE_CONSTANT(scheduled_state, "scheduled")
DEFINE_CONSTANT(started_state, "started")
DEFINE_CONSTANT(thread_id_key, "thread_id")
DEFINE_CONSTANT(thread_name_key, "thread_name")
#undef DEFINE_CONSTANT

// Whether to include stack trace in the Flight Recorder trace (default true)
inline const std::vector<std::string> TORCH_INCLUDE_STACK_TRACE = {
    "TORCH_INCLUDE_STACK_TRACE"};

// Whether to include only active collectives in the Flight Recorder trace
// (default false)
inline const std::vector<std::string> TORCH_INCLUDE_ONLY_ACTIVE = {
    "TORCH_INCLUDE_ONLY_ACTIVE"};

// Write NCCL debug info to local disk or any storage users define.
// There are some constrains we set for the debug info writer:
// 1. The writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recommended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter() = default;
  virtual void write(const std::string& trace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  DebugInfoWriter(
      const std::string& namePrefix,
      int rank,
      bool enableDynamicFilename = false) {
    filename_ = c10::str(namePrefix, rank);
    enable_dynamic_filename_ = enableDynamicFilename;
    rank_ = rank;
  }
  std::string filename_;
  int rank_;
  bool enable_dynamic_filename_;

 private:
  // NOLINTNEXTLINE(facebook-hte-NonPodStaticDeclaration)
  static std::unique_ptr<DebugInfoWriter> writer_;
  // NOLINTNEXTLINE(facebook-hte-NonPodStaticDeclaration)
  static std::atomic<bool> hasWriterRegistered_;
};

class FlightRecorder {
 public:
  static FlightRecorder* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
    static FlightRecorder* instance = [] {
      auto max_entries = env_to_value("TORCHCOMM_FR_BUFFER_SIZE", 2000);
      auto capture_cpp_stack = env_to_value("TORCHCOMM_FR_CPP_STACK", false);
      return new FlightRecorder(max_entries, capture_cpp_stack);
    }();
    return instance;
  }
  FlightRecorder(int64_t max_entries, bool capture_cpp_stack) {
    max_entries_ = max_entries;
    capture_cpp_stack_ = capture_cpp_stack;
    enabled_ = max_entries_ > 0;
    reset_epoch_start_idx_[0] = 0;
  }
  struct Entry {
    size_t id_{0}; // incremented id in the trace buffer
                   // used to figure out where in the circular entries
                   // buffer this entry will be located to
                   // update state information
    size_t reset_epoch_{0}; // epoch when this entry was created
    size_t pg_id_{0};
    std::tuple<std::string, std::string> pg_name_{}; // <group_name, group_desc>

    // collective_seq_id refers to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_{0};
    size_t op_id_{0};
    std::string profiling_name_{};

    std::shared_ptr<torch::CapturedTraceback> traceback_{};
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    c10::Event* start_{nullptr};
    c10::Event* end_{nullptr};

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_{0};

    // configured timeout for this entry
    c10::time_t timeout_ms_{0};

    std::optional<float> duration_{};

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on CUDA APIs.
    std::optional<c10::time_t> time_discovered_started_{};

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually completed, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_{};

    // size information for input/output tensors
    c10::SmallVector<int64_t, 4> input_dims_{};
    std::vector<c10::ScalarType> input_dtypes_{};
    c10::SmallVector<int64_t, 4> output_dims_{};
    std::vector<c10::ScalarType> output_dtypes_{};
    c10::SmallVector<int64_t, 8> sizes_{}; // flattened from inputs, outputs
    std::thread::id thread_id_{};
    std::string thread_name_{};
    bool retired_{false}; // a retired but not completed event has timed out

    // Returns the traceback of current entry, in string form.
    // Note: `getTraceback` invokes `torch::symbolize`, which may need to
    // acquire the GIL. If you don't want to block the current thread or take
    // the risk of a GIL deadlock, you can use an asynchronous calling mechanism
    // like std::async.
    std::string getTraceback();
  };

  void record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      c10::Event* start,
      c10::Event* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status);

  void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks);

  void record_accelerator_version(std::string comm_lib_version);

  std::vector<Entry> dump_entries();

  /*
  Mark an Event as completed and free its events.
  This is called by the watchdog thread, and is asynchronous from the
  perspective of the main thread.
  compute_duration defaults to true since retire_id is only called in the
  watchdog thread, which is currently a place we call cuda APIs which may hang,
  but care should be taken to avoid computing duration in any function that must
  never hang. (timing must also be enabled for compute_duration - see
  TORCH_NCCL_ENABLE_TIMING).
  */
  void retire_id(std::optional<size_t> id, bool compute_duration = true);

  void reset_all();

  std::string dump_json(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& extraDumpMap,
      bool includeCollectives,
      bool onlyActive);

  std::string dump(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& extraDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive);

  /**
   * Get the current number of entries.
   */
  size_t size() const;

 private:
  // Returns the entry with the given id and reset_epoch, if it exists.
  // Otherwise, returns std::nullopt.
  std::optional<Entry> getEntry(
      std::optional<size_t> id,
      std::optional<size_t> reset_epoch);

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive);

  // dump pg_entries
  const c10::Dict<c10::IValue, c10::IValue> getPgConfig();

  const std::map<std::string, std::map<std::string, std::string>>
  getPgConfigJson();

  // dump pg_status
  const c10::Dict<c10::IValue, c10::IValue> getPgStatus();

  const std::map<std::string, std::map<std::string, std::string>>
  getPgStatusJson();

  void update_state(Entry& r);

  // Returns the index in entries_ for the given id and reset_epoch.
  // Caller must hold mutex_lock before calling this method.
  size_t getIdxFromId(size_t id, size_t reset_epoch) const;

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t id_ = 0;
  size_t latest_op_id_ = 0; // tracks the latest op_id seen for buffer position
  size_t reset_epoch_ = 0;
  std::unordered_map<size_t, size_t>
      reset_epoch_start_idx_; // maps reset_epoch to the idx where it starts
  size_t collective_seq_id_{0};
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_;
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_;
  std::string comm_lib_version_;
  // Map from op_id to (id_, reset_epoch_) to pass correct values when retiring
  std::unordered_map<size_t, std::pair<size_t, size_t>> op_id_to_id_and_epoch_;
};

// ============================================================================
// TorchComm hooks
// ============================================================================

/**
 * FlightRecorderHook integrates FlightRecorder with TorchComm using hooks.
 * It uses the pre/post hook mechanism from TorchComm to record operations
 * and their states into the FlightRecorder.
 */
class FlightRecorderHook
    : public std::enable_shared_from_this<FlightRecorderHook> {
 public:
  /**
   * Create a FlightRecorderHook with the specified buffer size.
   * @param max_entries Maximum number of entries to keep in the ring buffer.
   *                    Older entries are overwritten when the buffer is full.
   * @param isolated If true, creates an isolated FlightRecorder instance
   *                 for this hook instead of using the global singleton.
   */
  explicit FlightRecorderHook(size_t max_entries = 2048, bool isolated = false);

  ~FlightRecorderHook();

  // Disable copy and move
  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  /**
   * Register this hook with a TorchComm communicator.
   * @param comm The communicator to register with.
   */
  void registerWithComm(std::shared_ptr<TorchComm> comm);

  /**
   * Dump all entries as a JSON string in the OSS FlightRecorder format.
   * This format is compatible with the fr_trace analyzer tools.
   * @param include_completed If false, only return entries that are not
   * completed.
   */
  std::string dump_json(bool include_completed = true) const;

  /**
   * Dump the flight recorder trace and write it to the debug info
   * writer.
   * @param rank The rank to use for the debug info writer.
   * @param include_completed If false, only dump entries that are not
   * completed.
   */
  void dump_file(int rank, bool include_completed = true) const;

  /**
   * Clear all entries and reset sequence counters.
   */
  void reset();

  /**
   * Check if the hook is enabled (has registered communicators).
   */
  bool isEnabled() const;

  /**
   * Get the current number of entries.
   */
  size_t size() const;

  /**
   * Get the underlying FlightRecorder instance.
   */
  FlightRecorder* getRecorder() {
    return recorder_;
  }

  /**
   * Hook called when a communicator is split.
   * Registers the new communicator with the flight recorder.
   * @param new_comm The newly created communicator from the split operation.
   */
  void splitHook(std::shared_ptr<TorchComm> new_comm);

 private:
  // Hook callback
  void onPreHook(
      const std::string& comm_name,
      size_t pg_id,
      const std::string& pg_desc,
      size_t op_id,
      const PreHookArgs& args);

  void onPostHook(size_t op_id, const PostHookArgs& args);

  FlightRecorder* recorder_;
  bool owns_recorder_{false}; // True when using isolated instance

  struct CommRegistration {
    std::weak_ptr<TorchComm> comm;
    size_t pg_id;
    std::string pg_desc;

    CommRegistration(std::weak_ptr<TorchComm> c, size_t id, std::string desc)
        : comm(std::move(c)), pg_id(id), pg_desc(std::move(desc)) {}
  };
  std::vector<CommRegistration> registrations_;
  bool enabled_{false};
};

} // namespace fr
} // namespace comms
} // namespace torch
