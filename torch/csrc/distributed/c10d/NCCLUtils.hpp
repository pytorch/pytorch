#pragma once

#ifdef USE_C10D_NCCL

#include <sched.h>
#include <cstdio>
#include <cstdlib>

#include <memory>
#include <mutex>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <optional>

constexpr int64_t kCommInitBusyWaitMillis = 2;

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

// Error out if (current time - startTime) is greater than timeout (sec).
#define C10D_CHECK_TIMEOUT(startTime, timeout)                              \
  do {                                                                      \
    auto currentTime = std::chrono::steady_clock::now();                    \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                           currentTime - startTime)                         \
                           .count();                                        \
    if (timeElapsed > timeout) {                                            \
      std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__);                                         \
      TORCH_CHECK_WITH(DistBackendError, false, err);                       \
    }                                                                       \
  } while (0)

// Macro to throw on a non-successful NCCL return value, non-blocking.
#define C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)      \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = nccl_nonblocking_timeout();                                \
    while (result == ncclInProgress) {                                        \
      C10D_CHECK_TIMEOUT(startTimepoint, timeout);                            \
      yield_fn;                                                               \
      ncclCommGetAsyncError(comm, &result);                                   \
    }                                                                         \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Sleep for kCommInitBusyWaitMillis milliseconds.
#define C10D_SCHED_SLEEP()     \
  std::this_thread::sleep_for( \
      std::chrono::milliseconds(kCommInitBusyWaitMillis))

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sched_yield() to yield the CPU.
// Thus suitable for NCCL calls that would quickly turn ncclSuccess, e.g.
// collectives.
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, sched_yield())

// Macro to throw exception on a non-successful NCCL return value or timeout.
// This macro uses sleep to yield the CPU.
// Thus suitable for NCCL calls that would take longer to turn ncclSuccess, e.g.
// ncclCommInitRankConfig, ncclCommFinalize, etc.
#define C10D_NCCL_CHECK_TIMEOUT_SLEEP(cmd, comm, failureReason) \
  C10D_NCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, C10D_SCHED_SLEEP())

#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  do {                                                                       \
    ncclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = nccl_nonblocking_timeout();                               \
    if (state == ncclInProgress) {                                           \
      do {                                                                   \
        C10D_CHECK_TIMEOUT(startTimepoint, timeout);                         \
        sched_yield();                                                       \
        ncclCommGetAsyncError(comm->getNcclComm(), &state);                  \
      } while (state == ncclInProgress);                                     \
    }                                                                        \
    if (state != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) + \
          "\n" + getNcclErrorDetailStr(state, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                        \
    }                                                                        \
  } while (0)

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
TORCH_API std::string getNcclVersion();
TORCH_API std::string ncclGetErrorWithVersion(ncclResult_t error);
int nccl_nonblocking_timeout();

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
TORCH_API std::string getNcclErrorDetailStr(
    ncclResult_t error,
    std::optional<std::string> processGroupFailureReason = std::nullopt);

// RAII wrapper for NCCL communicator
class NCCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit NCCLComm(ncclComm_t ncclComm) : ncclComm_(ncclComm) {}

  NCCLComm() = default;

  ~NCCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    LockType lock(mutex_);
    if (ncclComm_ && initialized_ && !aborted_) {
      at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
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
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex) {
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<NCCLComm>();
    C10D_NCCL_CHECK(
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank),
        std::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    comm->initialized_ = true;
    // Old style comm is always blocking.
    comm->nonBlocking_ = false;
    return comm;
  }

#ifdef NCCL_HAS_COMM_NONBLOCKING
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config) {
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<NCCLComm>();
    comm->nonBlocking_ = config.blocking == 0;
    LOG(INFO) << "Rank " << rank << ": creating NCCL communicator with mode: "
              << (comm->nonBlocking_ ? "nonblocking" : "blocking");
    C10D_NCCL_CHECK_NONBLOCKING(
        ncclCommInitRankConfig(
            &(comm->ncclComm_), numRanks, commId, rank, &config),
        std::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    // Under blocking mode, comm is initialized immediately after NCCL init
    // returns; Under nonblocking mode, we check whether comm is initialized the
    // *next* time ncclComm_ is accessed.
    comm->initialized_ = !comm->nonBlocking_;
    return comm;
  }

  static std::shared_ptr<NCCLComm> split(
      NCCLComm* source,
      int color_id,
      int rank,
      ncclConfig_t& config,
      std::vector<uint64_t>& ranks_ull);
#endif

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
  // NOLINTNEXTLINE(*-noexcept-move-*)
  NCCLComm(NCCLComm&& other) {
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    LockType lock(other.mutex_);
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
    std::swap(initialized_, other.initialized_);
    std::swap(nonBlocking_, other.nonBlocking_);
    std::swap(deviceIndex_, other.deviceIndex_);
  }

  ncclComm_t getNcclComm();

  std::optional<std::string> getNcclCommFailureReason() const {
    LockType lock(mutex_);
    return commFailureReason_;
  }

  void abort(std::optional<std::string> commFailureReason = std::nullopt) {
    LockType lock(mutex_);
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (aborted_ && !initialized_) {
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

  bool isInitialized() const {
    LockType lock(mutex_);
    return initialized_;
  }

  bool isAborted() const {
    LockType lock(mutex_);
    return aborted_;
  }

  uint64_t getCommSplitCounter() const {
    return ncclCommSplitCounter_;
  }

  ncclResult_t checkForNcclError() {
    LockType lock(mutex_);
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
    LockType lock(mutex_);
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

    void* handle = nullptr;
    // Use getNcclComm to make sure comm is ready before calling nccl APIs
    auto comm = getNcclComm();
    C10D_NCCL_CHECK(
        ncclCommRegister(comm, ptr, size, &handle),
        c10::str(
            "Failed to register segment with ptr ",
            ptr,
            ", size ",
            size,
            " on ncclComm_ ",
            comm));
    registeredSegmentHandles_[ptr] = handle;
    return ncclSuccess;
#else
    return ncclInvalidUsage;
#endif
  }

  ncclResult_t deregisterSegment(void* ptr) {
    LockType lock(mutex_);
#ifdef NCCL_HAS_COMM_REGISTER
    TORCH_CHECK(
        registeredSegmentHandles_.count(ptr) == 1,
        "Segment with ptr ",
        ptr,
        " is not registered on ncclComm_ ",
        ncclComm_);

    void* handle = registeredSegmentHandles_[ptr];
    // Use getNcclComm to make sure comm is ready before calling nccl APIs
    auto comm = getNcclComm();
    C10D_NCCL_CHECK(
        ncclCommDeregister(comm, handle),
        c10::str(
            "Failed to deregister segment handle ",
            handle,
            ", with ptr ",
            ptr,
            " on ncclComm_ ",
            comm));
    registeredSegmentHandles_.erase(ptr);
    return ncclSuccess;
#else
    return ncclInvalidUsage;
#endif
  }

  std::string repr() const {
    return c10::str((void*)ncclComm_);
  }

  friend class ProcessGroupNCCL;

 protected:
  // Unique nccl_id for this communicator.
  ncclUniqueId ncclId_{};
  bool aborted_{false};
  uint64_t ncclCommSplitCounter_{0};
  ncclResult_t ncclAsyncErr_{ncclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_{};
  // Optional reason for communicator failure, provided by ProcessGroupNCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_{};
  bool initialized_{false};
  // Whether this communicator is using nonblocking mode. Recorded during comm
  // creation or split. For safety, we give a default value of true (more
  // protection).
  bool nonBlocking_{true};
  // Device index for which the NCCL comm is created
  at::DeviceIndex deviceIndex_{-1};
#ifdef NCCL_HAS_COMM_REGISTER
  // Stores handlers for tensors registered by NCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif

 private:
  ncclComm_t ncclComm_{nullptr};
};

// Helper that automatically cleans up premul sums.
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  ncclRedOpRAII(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII& operator=(const ncclRedOpRAII&) = delete;
  ncclRedOpRAII(ncclRedOpRAII&& tmp) noexcept : ncclRedOpRAII() {
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
  ncclRedOp_t op_{};
  ncclComm_t comm_{};
  bool premul_sum_ = false;
};

} // namespace c10d

#endif // USE_C10D_NCCL
