#pragma once

#ifdef USE_C10D_NCCL

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>
#include <thread>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <nccl.h>

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

TORCH_API size_t hashTensors(const std::vector<at::Tensor>& tensors);
std::string getNcclVersion();
std::string ncclGetErrorWithVersion(ncclResult_t error);
bool nccl_use_nonblocking();
int nccl_nonblocking_timeout();

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
std::string getNcclErrorDetailStr(
    ncclResult_t error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

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
  virtual ~DebugInfoWriter();
  virtual void write(const std::string& ncclTrace);
  static DebugInfoWriter& getWriter(int rank);
  static void registerWriter(std::unique_ptr<DebugInfoWriter> writer);

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
        commFailureReason_(c10::nullopt),
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
        c10::nullopt);
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
          c10::nullopt);
    } else {
      C10D_NCCL_CHECK(
          ncclCommInitRankConfig(
              &(comm->ncclComm_), numRanks, commId, rank, &config),
          c10::nullopt);
      // under blocking mode, comm is initialized after NCCL CHECK
      isInitialized = true;
    }
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->initialized_ = isInitialized;
    return comm;
  }
#endif

#ifdef NCCL_HAS_COMM_SPLIT
  static std::shared_ptr<NCCLComm> split(
      NCCLComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config) {
    auto comm = std::make_shared<NCCLComm>();
    C10D_NCCL_CHECK(
        ncclCommSplit(
            source->ncclComm_, color_id, rank, &(comm->ncclComm_), &config),
        c10::nullopt);
    ++source->ncclCommSplitCounter_;
    comm->rank_ = rank;
    return comm;
  }
#endif

#if defined(IS_NCCL_EXP) && defined(NCCL_COMM_DUMP)
  std::unordered_map<std::string, std::string> ncclCommDump() {
    std::unordered_map<std::string, std::string> dump;
    if (isAborted()) {
      LOG(INFO) << "Communicator was aborted before trying to dump its state.";
      return dump;
    }
    C10D_NCCL_CHECK(::ncclCommDump(ncclComm_, dump), c10::nullopt);
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

  c10::optional<std::string> getNcclCommFailureReason() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return commFailureReason_;
  }

  void ncclCommAbort(
      c10::optional<std::string> commFailureReason = c10::nullopt) {
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
  c10::optional<std::string> commFailureReason_;
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

} // namespace c10d

#endif // USE_C10D_NCCL
