#pragma once
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <optional>

namespace c10d {

#define DEFINE_CONSTANT(name, value) \
  static c10::IValue name = value;   \
  static std::string name##_str = value;
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

// Write NCCL debug info to local disk or any storage users define.
// There are some constraints we set for the debug info writer:
// 1. The writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recommended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class TORCH_API DebugInfoWriter {
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
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};

template <typename EventType>
struct FlightRecorder {
  static FlightRecorder<EventType>* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static FlightRecorder<EventType>* instance =
        new FlightRecorder<EventType>();
    return instance;
  }
  FlightRecorder() {
    // NOTE: This default value (2000) is duplicated in ProcessGroupNCCL.cpp
    // and ProcessGroupNCCL.hpp because they cannot directly query max_entries_
    // (no public accessor). Keep these values in sync.
    max_entries_ = getCvarInt(
        {"TORCH_FR_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE"}, 2000);
    capture_cpp_stack_ = getCvarBool(
        {"TORCH_FR_CPP_STACK", "TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
    reset_epoch_start_idx_[0] = 0;
  }
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
    size_t reset_epoch_; // epoch when this entry was created
    size_t pg_id_;
    std::tuple<std::string, std::string> pg_name_; // <group_name, group_desc>

    // collective_seq_id and p2p_seq_id refer to actual kernel launches (e.g. 1
    // per coalesced group).
    // collective_seq_id only increments for true collective operations (over
    // all ranks in the group). p2p_seq_id only increments over non-collective
    // operations in the group. op_id refers to logical operations (e.g. one per
    // op inside coalesced group)
    size_t collective_seq_id_;
    size_t p2p_seq_id_;
    size_t op_id_;
    std::string profiling_name_;

    std::shared_ptr<torch::CapturedTraceback> traceback_;
    // we borrow pointers to start_ and end_ so we can query the state
    // on reporting. However, once the event is completed, the call
    // to `complete` will clear these.
    EventType *start_, *end_;

    // timestamp when the entry was created, likely close to the time the work
    // was 'enqueued'- not necessarily started
    c10::time_t time_created_;

    // configured timeout for this entry
    c10::time_t timeout_ms_;

    // Is this a P2P event?
    bool isP2P_;

    std::optional<float> duration_;

    // timestamp when our CPU threads discovered that the kernel started.
    // will always be _after_ it actually started, and can be very late
    // if the watchdog thread got stuck on CUDA APIs.
    std::optional<c10::time_t> time_discovered_started_;

    // timestamp when our CPU threads discovered that the kernel completed.
    // will always be _after_ it actually completed, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int64_t, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int64_t, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    std::thread::id thread_id_;
    std::string thread_name_;
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out

    // Returns the traceback of current entry, in string form.
    // Note: `getTraceback` invokes `torch::symbolize`, which may need to
    // acquire the GIL. If you don't want to block the current thread or take
    // the risk of a GIL deadlock, you can use an asynchronous calling mechanism
    // like std::async.
    TORCH_API std::string getTraceback();
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  size_t reset_epoch_ = 0;
  std::unordered_map<size_t, size_t>
      reset_epoch_start_idx_; // maps reset_epoch to the idx where it starts
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_;
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_;
  std::string comm_lib_version_;
  // Global rank of this process, or -1 if nothing has set it. The recorder is
  // a process-wide singleton and DebugInfoWriter names its output file
  // <prefix><rank>, so a dump triggered from somewhere that has no process
  // group in hand (the control plane) needs the rank from here.
  std::atomic<int> rank_{-1};

  void setRank(int rank) {
    rank_.store(rank, std::memory_order_relaxed);
  }

  int getRank() const {
    return rank_.load(std::memory_order_relaxed);
  }

  struct TraceIdentifier {
    std::optional<size_t> id;
    std::optional<size_t> reset_epoch;
  };

  TraceIdentifier recordWithResetEnabled(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      EventType* start,
      EventType* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status,
      bool isP2P);

  std::optional<size_t> record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      EventType* start,
      EventType* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status,
      bool isP2P);

  TORCH_API void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks);

  void record_accelerator_version(const std::string comm_lib_version);

  void update_state(Entry& r);

  std::vector<Entry> dump_entries();

  // Returns the index in entries_ for the given id and reset_epoch.
  // Caller must hold mutex_lock before calling this method.
  size_t getIdxFromId(size_t id, size_t reset_epoch) const;

  // Returns the entry with the given id and reset_epoch, if it exists.
  // Otherwise, returns std::nullopt.
  TORCH_API std::optional<Entry> getEntry(
      std::optional<size_t> id,
      std::optional<size_t> reset_epoch);

  TORCH_API std::optional<Entry> getEntry(std::optional<size_t> id);

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
  TORCH_API void retire_id(
      std::optional<size_t> id,
      std::optional<size_t> reset_epoch,
      bool compute_duration = true);

  TORCH_API void retire_id(
      std::optional<size_t> id,
      bool compute_duration = true);

  /*
  Retire an entry whose collective has been observed to complete by some means
  other than the entry's own events -- the c10d hooks own no events and instead
  wait on the op's Work. Marks the entry started and completed as of now (a
  collective that finished necessarily started) so a dump reports it
  "completed", and takes duration from the caller, which got it from the
  backend (Work::getDuration). duration is nullopt when the backend cannot
  report one, in which case the entry simply carries no duration_ms rather than
  a host-clock stand-in, which would measure how late the observation was.
  No-op if the ring buffer has already overwritten the entry.
  */
  TORCH_API void retire_completed(
      std::optional<size_t> id,
      std::optional<size_t> reset_epoch,
      std::optional<float> duration);

  TORCH_API void reset_all();

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
};

// Whether to include stack trace in the Flight Recorder trace (default true)
static std::vector<std::string> TORCH_INCLUDE_STACK_TRACE = {
    "TORCH_INCLUDE_STACK_TRACE"};

// Whether to include only active collectives in the Flight Recorder trace
// (default false)
static std::vector<std::string> TORCH_INCLUDE_ONLY_ACTIVE = {
    "TORCH_INCLUDE_ONLY_ACTIVE"};

// Whether to dump the trace to disk when a collective times out or fails
// (default true). The TORCH_NCCL_ alias is what stock ProcessGroupNCCL reads
// and what existing users set, so it must keep working.
static std::vector<std::string> TORCH_FR_DUMP_ON_TIMEOUT = {
    "TORCH_FR_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DUMP_ON_TIMEOUT"};

// How long a single dump attempt on the failure path may take before it is
// abandoned (default 15s), same knob and same default as stock
// ProcessGroupNCCL's heartbeat monitor, which bounds its own dump with it.
static std::vector<std::string> TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_FR_WAIT_TIMEOUT_DUMP_MILSEC",
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Backend name of the default FlightRecorder<c10::Event> instance, i.e. the
// process-wide singleton FlightRecorder<c10::Event>::get() returns. That is
// the instance ProcessGroupGloo records into natively, so leaving the backend
// unnamed keeps every dump API answering exactly what it always has.
constexpr const char* kDefaultFRBackend = "gloo";

// Per-backend FlightRecorder<c10::Event> instances, created on first use.
// Every hooked backend records into its own ring buffer, so a chatty backend
// cannot evict another's entries and their pg_ids cannot collide. The built-in
// backends already have that property -- gloo in this c10::Event instance,
// nccl in the CUDAEvent one -- and this restores it for hooked backends.
//
// The registry lives in libtorch_cpu and is only reachable through this
// exported function. FlightRecorder<EventType>::get()'s function-local static
// is not an exported symbol, so any other shared library that instantiates the
// template gets a private, empty instance of its own; going through
// getFlightRecorder keeps every caller on the one that holds the trace.
TORCH_API FlightRecorder<c10::Event>* getFlightRecorder(
    const std::string& backend);

// Whether a backend writes to a FlightRecorder of its own, in which case the
// c10d hooks must not record its ops on top. Narrower than
// _FR_SELF_RECORDING_BACKENDS in distributed_c10d.py, which also lists
// backends that never communicate.
TORCH_API bool recordsFlightRecorderNatively(const std::string& backend);

// Dumps the fr traces and additional information about the Process
// Group.
TORCH_API std::string dump_fr_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend = kDefaultFRBackend);

// Dumps the fr traces and additional information about the Process
// Group in JSON formatted string.
// We don't include stack traces in JSON format as it is far too much data.
TORCH_API std::string dump_fr_trace_json(
    bool includeCollectives,
    bool onlyActive,
    const std::string& backend = kDefaultFRBackend);

// Dumps the fr traces to the file the registered DebugInfoWriter points at
// (<TORCH_FR_DUMP_TEMP_FILE><rank> by default), in the same pickled format as
// dump_fr_trace. This is the backend-agnostic counterpart of
// ProcessGroupNCCL::dumpDebuggingInfo.
TORCH_API void dump_fr_trace_file(
    int rank,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend = kDefaultFRBackend);

// dump_fr_trace_file for the rank the recorder was told about at hook attach.
// Returns false without writing anything if the recorder is off or no rank was
// ever set, which also means nothing was recorded. Used by
// FlightRecorderHook's abort hook, whose whole job is a best-effort dump on a
// failure that may have happened before any group was attached.
//
// Callers outside libtorch_cpu must also use this instead of reaching into
// FlightRecorder<c10::Event>::get() themselves: get()'s function-local static
// is not an exported symbol, so every shared library that instantiates the
// template ends up with a private, empty recorder of its own. Only code linked
// into libtorch_cpu -- where the hooks and the gloo backend record -- sees the
// instance that actually holds the trace.
TORCH_API bool try_dump_fr_trace_file(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend = kDefaultFRBackend);

// Drops everything recorded so far, so a subsequent dump only shows what came
// after. Backend-agnostic counterpart of reset_nccl_trace. Same cross-DSO
// caveat as try_dump_fr_trace_file: callers outside libtorch_cpu must go
// through this instead of FlightRecorder<c10::Event>::get().
TORCH_API void reset_fr_trace(const std::string& backend = kDefaultFRBackend);
} // namespace c10d
