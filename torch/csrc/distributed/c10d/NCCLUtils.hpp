#pragma once

#ifdef USE_C10D_NCCL

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>
#include <thread>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <optional>

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 14)
#define NCCL_HAS_COMM_NONBLOCKING
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 18)
#define NCCL_HAS_COMM_SPLIT
#endif

// ncclGetLastError() is enabled only for NCCL versions 2.13+
// ncclRemoteError only exists in NCCL versions 2.13+
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 13)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_GET_LAST_ERROR
#define NCCL_REMOTE_ERROR
#endif

// Error checking is enabled only for NCCL versions 2.4+ since ncclCommAbort()
// and ncclCommGetAsyncError() are not supported in earlier versions.
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 4)
#define ENABLE_NCCL_ERROR_CHECKING
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_ERROR_CHECKING
#endif

// P2P is enabled only for NCCL versions 2.7+ since ncclSend()
// and ncclRecv() are not supported in earlier versions.
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 7)
#define ENABLE_NCCL_P2P_SUPPORT
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_P2P_SUPPORT
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 11)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 17)
#define NCCL_HAS_COMM_CTA_CGA
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_CTA_CGA
#endif

#if defined(NCCL_REGISTRATION_SUPPORTED) ||                              \
    ((defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
      (NCCL_MINOR >= 19)))
#define NCCL_HAS_COMM_REGISTER
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_REGISTER
#endif

// Macro to throw on a non-successful NCCL return value.
#define C10D_NCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Macro to throw on a non-successful NCCL return value for NONBLOCKING calls.
#define C10D_NCCL_CHECK_NONBLOCKING(cmd, failureReason)                       \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess && result != ncclInProgress) {                  \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Macro to throw on a non-successful NCCL return value, non-blocking.
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason)                     \
  ncclResult_t result = cmd;                                                  \
  auto startTimepoint = std::chrono::steady_clock::now();                     \
  while (result == ncclInProgress) {                                          \
    if (nccl_nonblocking_timeout() > 0) {                                     \
      auto currentTimepoint = std::chrono::steady_clock::now();               \
      auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                             currentTimepoint - startTimepoint)               \
                             .count();                                        \
      if (timeElapsed > nccl_nonblocking_timeout()) {                         \
        std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ", " +                                 \
            ncclGetErrorWithVersion(result) + "\n" +                          \
            getNcclErrorDetailStr(result, failureReason);                     \
        TORCH_CHECK_WITH(DistBackendError, false, err);                       \
      }                                                                       \
    }                                                                         \
    ncclCommGetAsyncError(comm, &result);                                     \
  }                                                                           \
  if (result != ncclSuccess) {                                                \
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +       \
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +   \
        "\n" + getNcclErrorDetailStr(result, failureReason);                  \
    TORCH_CHECK_WITH(DistBackendError, false, err);                           \
  }

#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  ncclResult_t state = cmd;                                                  \
  auto startTimepoint = std::chrono::steady_clock::now();                    \
  if (state == ncclInProgress) {                                             \
    do {                                                                     \
      if (nccl_nonblocking_timeout() > 0) {                                  \
        auto currentTimepoint = std::chrono::steady_clock::now();            \
        auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>( \
                               currentTimepoint - startTimepoint)            \
                               .count();                                     \
        if (timeElapsed > nccl_nonblocking_timeout()) {                      \
          std::string err = "NCCL timeout in: " + std::string(__FILE__) +    \
              ":" + std::to_string(__LINE__) + ", " +                        \
              ncclGetErrorWithVersion(state) + "\n" +                        \
              getNcclErrorDetailStr(state, failureReason);                   \
          TORCH_CHECK_WITH(DistBackendError, false, err);                    \
        }                                                                    \
      }                                                                      \
      ncclCommGetAsyncError(comm->getNcclComm(), &state);                    \
    } while (state == ncclInProgress);                                       \
  }                                                                          \
  if (state != ncclSuccess) {                                                \
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +      \
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) +   \
        "\n" + getNcclErrorDetailStr(state, failureReason);                  \
    TORCH_CHECK_WITH(DistBackendError, false, err);                          \
  }

// Macro to print and abort on a non-successful NCCL return value.
#define C10D_NCCL_ASSERT(cmd)                            \
  do {                                                   \
    ncclResult_t result = cmd;                           \
    if (result != ncclSuccess) {                         \
      std::string err = ncclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "NCCL error in: %s:%d, %s\n",                  \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

namespace c10d {
#define DEFINE_CONSTANT(name, value) \
  static c10::IValue name = value;   \
  static std::string name##_str = value;
DEFINE_CONSTANT(entries_key, "entries");
DEFINE_CONSTANT(nccl_comm_key, "nccl_comm_state");
DEFINE_CONSTANT(version_key, "version");
// Update whenever changing contents or formatting of the dump
// (minor when adding fields, major when changing existing fields)
// Also update both JSON and Pickle dumps to make use of the newly defined
// field(s).
DEFINE_CONSTANT(version_val, "2.3");
DEFINE_CONSTANT(pg_config_key, "pg_config");
DEFINE_CONSTANT(pg_status_key, "pg_status");
DEFINE_CONSTANT(record_id_key, "record_id");
DEFINE_CONSTANT(pg_id_key, "pg_id");
DEFINE_CONSTANT(pg_name_key, "process_group");
DEFINE_CONSTANT(collective_seq_id_key, "collective_seq_id");
DEFINE_CONSTANT(p2p_seq_id_key, "p2p_seq_id");
DEFINE_CONSTANT(is_p2p_key, "is_p2p");
DEFINE_CONSTANT(op_id_key, "op_id");
DEFINE_CONSTANT(profiling_name_key, "profiling_name");
DEFINE_CONSTANT(input_sizes_key, "input_sizes");
DEFINE_CONSTANT(input_dtypes_key, "input_dtypes");
DEFINE_CONSTANT(output_sizes_key, "output_sizes");
DEFINE_CONSTANT(output_dtypes_key, "output_dtypes");
DEFINE_CONSTANT(time_created_key, "time_created_ns");
DEFINE_CONSTANT(duration_key, "duration_ms");
DEFINE_CONSTANT(timeout_key, "timeout_ms");
DEFINE_CONSTANT(frames_key, "frames");
DEFINE_CONSTANT(state_key, "state");
DEFINE_CONSTANT(line_key, "line");
DEFINE_CONSTANT(name_key, "name");
DEFINE_CONSTANT(filename_key, "filename");
DEFINE_CONSTANT(retired_key, "retired");
DEFINE_CONSTANT(time_discovered_started_key, "time_discovered_started_ns");
DEFINE_CONSTANT(time_discovered_completed_key, "time_discovered_completed_ns");
DEFINE_CONSTANT(completed_state, "completed");
DEFINE_CONSTANT(scheduled_state, "scheduled");
DEFINE_CONSTANT(started_state, "started");
#undef DEFINE_CONSTANT

TORCH_API size_t hashTensors(const std::vector<at::Tensor>& tensors);
TORCH_API std::string getNcclVersion();
TORCH_API std::string ncclGetErrorWithVersion(ncclResult_t error);
bool nccl_use_nonblocking();
int nccl_nonblocking_timeout();
bool shouldBroadcastNCCLUniqueID(bool isSendRecvSelf);

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
TORCH_API std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

// Write NCCL debug info to local disk or any storage users define.
// There are some constrains we set for the debug info writer:
// 1. The writer should only be registered once.
// 2. Once registered, users cannot change it including un-register.
// 3. It is recommended to register the customized writer in the trainer setup,
//    If users don't register before calling launchAsyncDebugDump, then users
//    lose the chance to register (and the default writer will be
//    auto-registered).
class TORCH_API DebugInfoWriter {
 public:
  virtual ~DebugInfoWriter() = default;
  virtual void write(const std::string& ncclTrace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);
  virtual std::string getWriterTarget() {
    return filename_;
  }

 protected:
  DebugInfoWriter(std::string namePrefix, int rank) {
    filename_ = c10::str(namePrefix, rank);
  }
  std::string filename_;

 private:
  static std::unique_ptr<DebugInfoWriter> writer_;
  static std::atomic<bool> hasWriterRegistered_;
};

// RAII wrapper for NCCL communicator
class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm),
        aborted_(false),
        ncclAsyncErr_(ncclSuccess),
        commFailureReason_(std::nullopt),
        initialized_(false) {}

  NCCLComm() : NCCLComm(nullptr) {}

  ~NCCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (ncclComm_ && !aborted_) {
#ifdef ENABLE_NCCL_ERROR_CHECKING
      // Use ncclCommAbort instead of ncclCommDestroy here since
      // ncclCommDestroy could block forever waiting for work to complete on
      // the communicator.
      C10D_NCCL_ASSERT(::ncclCommAbort(ncclComm_));
#else
      C10D_NCCL_ASSERT(::ncclCommDestroy(ncclComm_));
#endif
    }
  }

  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();
    C10D_NCCL_CHECK(
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
        std::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->initialized_ = true;
    return comm;
  }

#ifdef NCCL_HAS_COMM_NONBLOCKING
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      ncclConfig_t& config) {
    auto comm = std::make_shared<NCCLComm>();
    bool isInitialized = false;
    if (nccl_use_nonblocking()) {
      config.blocking = 0;
      LOG(INFO) << "Rank " << rank
                << ": creating NCCL communicator in nonblocking mode";
      C10D_NCCL_CHECK_NONBLOCKING(
          ncclCommInitRankConfig(
              &(comm->ncclComm_), numRanks, commId, rank, &config),
          std::nullopt);
    } else {
      C10D_NCCL_CHECK(
          ncclCommInitRankConfig(
              &(comm->ncclComm_), numRanks, commId, rank, &config),
          std::nullopt);
      // under blocking mode, comm is initialized after NCCL CHECK
      isInitialized = true;
    }
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->initialized_ = isInitialized;
    return comm;
  }
#endif

  static std::shared_ptr<NCCLComm> split(
      NCCLComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config,
      std::vector<uint64_t>& ranks_ull);

#if defined(IS_NCCLX) && defined(NCCL_COMM_DUMP)
  std::unordered_map<std::string, std::string> ncclCommDump() {
    std::unordered_map<std::string, std::string> dump;
    if (isAborted()) {
      LOG(INFO) << "Communicator was aborted before trying to dump its state.";
      return dump;
    }
    C10D_NCCL_CHECK(::ncclCommDump(ncclComm_, dump), std::nullopt);
    return dump;
  }
#endif

  ncclUniqueId getNcclId() {
    return ncclId_;
  }

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  NCCLComm& operator=(NCCLComm&& other) = delete;

  // Move constructable
  NCCLComm(NCCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
    std::swap(initialized_, other.initialized_);
  }

  ncclComm_t getNcclComm();

  std::optional<std::string> getNcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void ncclCommAbort(
      std::optional<std::string> commFailureReason = std::nullopt) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (aborted_) {
      // Should not abort twice.
      return;
    }

#ifdef NCCL_HAS_COMM_REGISTER
    // Deregister all registered segments before aborting.
    for (auto& it : registeredSegmentHandles_) {
      void* handle = it.second;
      C10D_NCCL_CHECK(
          ::ncclCommDeregister(ncclComm_, handle),
          c10::str(
              "Failed to deregister segment handle ",
              handle,
              " on ncclComm_ ",
              ncclComm_));
    }
    registeredSegmentHandles_.clear();
#endif

    // Set true failure reason if provided by ProcessGroupNCCL (e.g. work
    // timeout)
    commFailureReason_ = commFailureReason;
    LOG(INFO) << "Aborting ncclComm_ " << ncclComm_ << " with reason: "
              << (commFailureReason ? *commFailureReason
                                    : "No abort reason provided.");
#ifndef NCCL_HAS_COMM_NONBLOCKING
    C10D_NCCL_CHECK(::ncclCommAbort(ncclComm_), commFailureReason_);
#else
    C10D_NCCL_CHECK_TIMEOUT(
        ::ncclCommAbort(ncclComm_), ncclComm_, commFailureReason_);
#endif
    aborted_ = true;
    ncclComm_ = nullptr;

    // Set an appropriate error so that we avoid using the communicator.
    if (ncclAsyncErr_ == ncclSuccess) {
      ncclAsyncErr_ = ncclSystemError;
    }
#else
    // This is a NOOP, if error checks are disabled.
    return;
#endif
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

  uint64_t getCommSplitCounter() const {
    return ncclCommSplitCounter_;
  }

  ncclResult_t checkForNcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (ncclAsyncErr_ != ncclSuccess) {
      return ncclAsyncErr_;
    }
    C10D_NCCL_CHECK(
        ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), commFailureReason_);
    return ncclAsyncErr_;
#else
    // Always return success, if error checks are disabled.
    return ncclSuccess;
#endif
  }

  ncclResult_t registerSegment(void* ptr, size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
    // We register only segments from cache allocator
    // which are guaranteed to be with disjoint addr ranges. Thus, a ptr always
    // maps to a unique handle and should not be registered before the current
    // ptr is deregistered and freed.
    TORCH_CHECK(
        registeredSegmentHandles_.count(ptr) == 0,
        "Segment with ptr ",
        ptr,
        " has already been registered on ncclComm_ ",
        ncclComm_);

    void* handle;
    C10D_NCCL_CHECK(
        ncclCommRegister(ncclComm_, ptr, size, &handle),
        c10::str(
            "Failed to register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on ncclComm_ ",
            ncclComm_));
    registeredSegmentHandles_[ptr] = handle;
    return ncclSuccess;
#else
    return ncclInvalidUsage;
#endif
  }

  ncclResult_t deregisterSegment(void* ptr) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
    TORCH_CHECK(
        registeredSegmentHandles_.count(ptr) == 1,
        "Segment with ptr ",
        ptr,
        " is not registered on ncclComm_ ",
        ncclComm_);

    void* handle = registeredSegmentHandles_[ptr];
    C10D_NCCL_CHECK(
        ncclCommDeregister(ncclComm_, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on ncclComm_ ",
            ncclComm_));
    registeredSegmentHandles_.erase(ptr);
    return ncclSuccess;
#else
    return ncclInvalidUsage;
#endif
  }

  friend class ProcessGroupNCCL;

 protected:
  // a helper function to wait until the communicator is initialized;
  void waitUntilInitialized(int timeoutSecs);
  ncclComm_t ncclComm_;
  // Unique nccl_id for this communicator.
  ncclUniqueId ncclId_;
  bool aborted_;
  uint64_t ncclCommSplitCounter_{0};
  ncclResult_t ncclAsyncErr_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupNCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_;
  bool initialized_{false};
#ifdef NCCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by NCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif
};

// Helper that automatically cleans up premul sums.
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ncclRedOpRAII(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII& operator=(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII(ncclRedOpRAII&& tmp) : ncclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
#if defined(ENABLE_NCCL_PREMUL_SUM_SUPPORT)
  ~ncclRedOpRAII() {
    if (premul_sum_) {
      ncclRedOpDestroy(op_, comm_);
    }
  }
#endif
  operator ncclRedOp_t() const {
    return op_;
  }
  ncclRedOp_t op_;
  ncclComm_t comm_;
  bool premul_sum_ = false;
};

/* Helper used by work::getDuration() and nccl flight recorder */
float getDurationFromEvent(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent);

struct NCCLTraceBuffer {
  static NCCLTraceBuffer* get() {
    // intentionally leak on exit
    // because this will hold python state that may get destructed
    static NCCLTraceBuffer* instance = new NCCLTraceBuffer();
    return instance;
  }
  NCCLTraceBuffer() {
    max_entries_ = getCvarInt({"TORCH_NCCL_TRACE_BUFFER_SIZE"}, 0);
    capture_cpp_stack_ = getCvarBool({"TORCH_NCCL_TRACE_CPP_STACK"}, false);
    enabled_ = max_entries_ > 0;
  }
  using Event = at::cuda::CUDAEvent;
  struct Entry {
    size_t id_; // incremented id in the trace buffer
                // used to figure out where in the circular entries
                // buffer this entry will be located to
                // update state information
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
    Event *start_, *end_;

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
    // will always be _after_ it actually complated, and can be the same time
    // as the discovery of the start if the watchdog thread is stuck on CUDA
    // APIs
    std::optional<c10::time_t> time_discovered_completed_;

    // size information for input/output tensors
    c10::SmallVector<int, 4> input_dims_;
    std::vector<c10::ScalarType> input_dtypes_;
    c10::SmallVector<int, 4> output_dims_;
    std::vector<c10::ScalarType> output_dtypes_;
    c10::SmallVector<int64_t, 8> sizes_; // flattened from inputs, outputs
    bool retired_ = false; // is this work entry no longer in the workMetaList_?
                           // a retired but not completed event has timed out
  };

  bool enabled_ = false;
  bool capture_cpp_stack_ = false;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  size_t max_entries_ = 0;
  size_t next_ = 0;
  size_t id_ = 0;
  std::map<size_t, std::shared_ptr<ProcessGroupStatus>> all_pg_status_ = {};
  std::map<std::tuple<std::string, std::string>, std::vector<uint64_t>>
      pg_name_to_ranks_ = {};

  std::optional<size_t> record(
      size_t pg_id,
      const std::tuple<std::string, std::string>& pg_name,
      size_t collective_seq_id,
      size_t p2p_seq_id,
      size_t op_id,
      std::string profiling_name,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs,
      Event* start,
      Event* end,
      std::chrono::milliseconds timeout_ms,
      std::shared_ptr<ProcessGroupStatus> pg_status,
      bool isP2P) {
    if (!enabled_) {
      return std::nullopt;
    }
    if (all_pg_status_.find(pg_id) == all_pg_status_.end()) {
      // Current pg_status is not in FR.
      all_pg_status_[pg_id] = pg_status;
    }
    auto traceback =
        torch::CapturedTraceback::gather(true, true, capture_cpp_stack_);
    std::lock_guard<std::mutex> guard(mutex_);

    auto te = Entry{
        id_,
        pg_id,
        pg_name,
        collective_seq_id,
        p2p_seq_id,
        op_id,
        std::move(profiling_name),
        std::move(traceback),
        std::move(start),
        std::move(end),
        c10::getTime(),
        timeout_ms.count(),
        isP2P,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        {},
        {},
        {},
        {},
        {},
        false};

    for (const auto& input : inputs) {
      c10::IntArrayRef sizes = input.sizes();
      te.input_dtypes_.push_back(input.dtype().toScalarType());
      te.input_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    for (const auto& output : outputs) {
      c10::IntArrayRef sizes = output.sizes();
      te.output_dtypes_.push_back(output.dtype().toScalarType());
      te.output_dims_.push_back(sizes.size());
      te.sizes_.insert(te.sizes_.end(), sizes.begin(), sizes.end());
    }

    if (entries_.size() < max_entries_) {
      entries_.emplace_back(std::move(te));
    } else {
      entries_[next_++] = std::move(te);
      if (next_ == max_entries_) {
        next_ = 0;
      }
    }
    return id_++;
  }

  void record_pg_ranks(
      const std::tuple<std::string, std::string>& pg_name,
      std::vector<uint64_t> ranks) {
    if (!enabled_) {
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    pg_name_to_ranks_[pg_name] = ranks;
  }

  void update_state(Entry& r) {
    if (r.start_ != nullptr) {
      bool started = r.start_->query();
      if (started && !r.time_discovered_started_) {
        r.time_discovered_started_ = c10::getTime();
      }
    }
    if (r.end_ != nullptr) {
      bool completed = r.end_->query();
      if (completed && !r.time_discovered_completed_) {
        r.time_discovered_completed_ = c10::getTime();
      }
    }
  }

  std::vector<Entry> dump_entries() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<Entry> result;
    result.reserve(entries_.size());
    result.insert(result.end(), entries_.begin() + next_, entries_.end());
    result.insert(result.end(), entries_.begin(), entries_.begin() + next_);
    // query any remaining events
    for (auto& r : result) {
      update_state(r);
      r.start_ = r.end_ = nullptr;
    }
    return result;
  }

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
  void retire_id(std::optional<size_t> id, bool compute_duration = true) {
    if (!enabled_ || !id) {
      return;
    }

    bool can_compute_duration = false;
    Event* startEvent = nullptr;
    Event* endEvent = nullptr;
    std::optional<float> duration = std::nullopt;

    std::unique_lock<std::mutex> guard(mutex_);

    Entry* entry = &entries_.at(*id % max_entries_);
    if (entry->id_ == *id) {
      update_state(*entry);

      if (compute_duration) {
        can_compute_duration = entry->time_discovered_completed_.has_value() &&
            entry->start_ && entry->end_;
        startEvent = entry->start_;
        endEvent = entry->end_;
      }
      entry->retired_ = true;
      entry->start_ = entry->end_ = nullptr;
    }

    if (can_compute_duration) {
      // Compute duration without without holding the lock, because
      // cudaEventDuration() can hang, and we need to acquire the lock before we
      // can dump(), which we never want to block.
      guard.unlock();
      duration = getDurationFromEvent(*startEvent, *endEvent);
      guard.lock();

      // Refresh the entry pointer, see if the entry has been overwritten
      entry = &entries_.at(*id % max_entries_);
      if (entry->id_ != *id) {
        LOG(INFO)
            << "retire_id abandoned for id " << *id
            << ", event was overwritten while waiting to compute duration.";
        return;
      }
      if (duration.has_value()) {
        entry->duration_ = duration.value();
      }
    }
  }

  const c10::List<c10::IValue> getCollectiveTrace(
      bool includeStacktraces,
      bool onlyActive) {
    auto entries = new_list();
    auto result = dump_entries();
    std::vector<torch::CapturedTraceback*> tracebacks;
    torch::SymbolizedTracebacks stracebacks;
    std::vector<c10::IValue> all_frames;
    if (includeStacktraces) {
      for (auto& e : result) {
        tracebacks.push_back(e.traceback_.get());
      }
      stracebacks = torch::symbolize(tracebacks);
      for (const auto& f : stracebacks.all_frames) {
        auto d = new_dict();
        d.insert(name_key, f.funcname);
        d.insert(filename_key, f.filename);
        d.insert(line_key, int64_t(f.lineno));
        all_frames.emplace_back(std::move(d));
      }
    }
    for (auto i : c10::irange(result.size())) {
      auto dict = new_dict();
      auto& e = result.at(i);
      // Skip completed events
      if (onlyActive && e.time_discovered_completed_.has_value()) {
        continue;
      }
      if (includeStacktraces) {
        auto& tb = stracebacks.tracebacks.at(i);
        auto frames = new_list();
        for (int64_t frame : tb) {
          frames.push_back(all_frames.at(frame));
        }
        dict.insert(frames_key, frames);
      }

      dict.insert(record_id_key, int64_t(e.id_));
      dict.insert(pg_id_key, int64_t(e.pg_id_));
      dict.insert(pg_name_key, e.pg_name_);
      dict.insert(collective_seq_id_key, int64_t(e.collective_seq_id_));
      dict.insert(p2p_seq_id_key, int64_t(e.p2p_seq_id_));
      dict.insert(op_id_key, int64_t(e.op_id_));
      dict.insert(profiling_name_key, e.profiling_name_);
      dict.insert(time_created_key, int64_t(e.time_created_));
      if (e.duration_) {
        dict.insert(duration_key, *e.duration_);
      }

      auto it = e.sizes_.begin();
      auto read_sizes = [&](const c10::SmallVector<int, 4>& dims) {
        auto sizes = new_list();
        for (auto dim : dims) {
          auto arg_sizes = new_list();
          for (C10_UNUSED auto i : c10::irange(dim)) {
            arg_sizes.push_back(*it++);
          }
          sizes.push_back(arg_sizes);
        }
        return sizes;
      };

      dict.insert(input_sizes_key, read_sizes(e.input_dims_));
      std::vector<std::string> input_dtypes_strs;
      input_dtypes_strs.reserve(e.input_dtypes_.size());
      for (const auto& input_dtype : e.input_dtypes_) {
        input_dtypes_strs.push_back(c10::toString(input_dtype));
      }
      dict.insert(input_dtypes_key, input_dtypes_strs);
      dict.insert(output_sizes_key, read_sizes(e.output_dims_));
      std::vector<std::string> output_dtypes_strs;
      output_dtypes_strs.reserve(e.output_dtypes_.size());
      for (const auto& output_dtype : e.output_dtypes_) {
        output_dtypes_strs.push_back(c10::toString(output_dtype));
      }
      dict.insert(output_dtypes_key, output_dtypes_strs);
      if (e.time_discovered_completed_.has_value()) {
        dict.insert(state_key, completed_state);
      } else if (e.time_discovered_started_.has_value()) {
        dict.insert(state_key, started_state);
      } else {
        dict.insert(state_key, scheduled_state);
      }

      dict.insert(
          time_discovered_started_key,
          e.time_discovered_started_.has_value()
              ? int64_t(*e.time_discovered_started_)
              : c10::IValue());
      dict.insert(
          time_discovered_completed_key,
          e.time_discovered_completed_.has_value()
              ? int64_t(*e.time_discovered_completed_)
              : c10::IValue());
      dict.insert(retired_key, e.retired_);
      dict.insert(timeout_key, e.timeout_ms_);
      dict.insert(is_p2p_key, e.isP2P_);

      entries.push_back(dict);
    }
    return entries;
  }

  // dump pg_entries
  const c10::Dict<c10::IValue, c10::IValue> getPgConfig() {
    auto pg_config = new_dict();
    for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
      auto pg_info = new_dict();
      pg_info.insert("name", std::get<0>(pg_name));
      pg_info.insert("desc", std::get<1>(pg_name));
      pg_info.insert("ranks", ranks_str(ranks));
      pg_config.insert(std::get<0>(pg_name), pg_info);
    }
    return pg_config;
  }

  const std::map<std::string, std::map<std::string, std::string>>
  getPgConfigJson() {
    std::map<std::string, std::map<std::string, std::string>> result;
    for (const auto& [pg_name, ranks] : pg_name_to_ranks_) {
      auto pg_info = std::map<std::string, std::string>();
      pg_info["name"] = std::get<0>(pg_name);
      pg_info["desc"] = std::get<1>(pg_name);
      pg_info["ranks"] = ranks_str(ranks);
      result.emplace(std::get<0>(pg_name), pg_info);
    }
    return result;
  }

  // dump pg_status
  const c10::Dict<c10::IValue, c10::IValue> getPgStatus() {
    auto all_pg_status = new_dict();
    for (const auto& [pg_id, status] : all_pg_status_) {
      auto pg_status = new_dict();
      pg_status.insert("last_enqueued_collective", status->lastEnqueuedSeq);
      pg_status.insert("last_started_collective", status->lastStartedSeq);
      pg_status.insert("last_completed_collective", status->lastCompletedSeq);
      all_pg_status.insert(std::to_string(pg_id), pg_status);
    }
    return all_pg_status;
  }

  const std::map<std::string, std::map<std::string, std::string>>
  getPgStatusJson() {
    std::map<std::string, std::map<std::string, std::string>> result;
    for (const auto& [pg_id, status] : all_pg_status_) {
      auto pg_status = std::map<std::string, std::string>();
      pg_status["last_enqueued_collective"] =
          std::to_string(status->lastEnqueuedSeq);
      pg_status["last_started_collective"] =
          std::to_string(status->lastStartedSeq);
      pg_status["last_completed_collective"] =
          std::to_string(status->lastCompletedSeq);
      result[std::to_string(pg_id)] = pg_status;
    }
    return result;
  }

  std::string dump_json(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& ncclDumpMap,
      bool includeCollectives,
      bool onlyActive);
  // dump all collectives + ncclDumpMap
  std::string dump(
      const std::optional<std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>>& ncclDumpMap,
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive) {
    auto result = new_dict();
    // common values
    result.insert(version_key, version_val);
    result.insert(pg_config_key, getPgConfig());
    result.insert(pg_status_key, getPgStatus());

    // collective trace
    if (includeCollectives) {
      result.insert(
          entries_key, getCollectiveTrace(includeStackTraces, onlyActive));
    }

    // convert ncclDumpMap into a dictionary
    auto per_comm_dict = new_dict();
    if (ncclDumpMap.has_value()) {
      for (const auto& [ncclId, ncclDump] : ncclDumpMap.value()) {
        auto inner_dict = new_dict();
        for (const auto& [key, value] : ncclDump) {
          inner_dict.insert(key, value);
        }
        per_comm_dict.insert(ncclId, inner_dict);
      }
    }
    if (per_comm_dict.size() > 0) {
      result.insert(nccl_comm_key, per_comm_dict);
    }
    return pickle_str(result);
  }
};
} // namespace c10d

#endif // USE_C10D_NCCL
