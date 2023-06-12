#pragma once

#ifdef USE_C10D_NCCL

#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <mutex>

#include <nccl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 14)
#define NCCL_HAS_COMM_NONBLOCKING
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

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 11)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && (NCCL_MINOR >= 17)
#define NCCL_HAS_COMM_CTA_CGA
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define NCCL_HAS_COMM_CTA_CGA
#endif

// Macro to throw on a non-successful NCCL return value.
#define C10D_NCCL_CHECK(cmd, failureReason)                                                  \
  do {                                                                        \
    ncclResult_t result = cmd;                                                \
    if (result != ncclSuccess) {                                              \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) + \
          "\n" + getNcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK_WITH(DistBackendError, false, err);                         \
    }                                                                         \
  } while (0)

// Macro to throw on a non-successful NCCL return value, non-blocking.
#define C10D_NCCL_CHECK_TIMEOUT(cmd, comm, failureReason)                               \
  ncclResult_t result = cmd;                                                                                          \
  auto startTimepoint = std::chrono::steady_clock::now();                                                             \
  while (result == ncclInProgress) {                                                                                  \
    if (nccl_nonblocking_timeout() > 0) {                                                                             \
      auto currentTimepoint = std::chrono::steady_clock::now();                                                       \
      auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTimepoint - startTimepoint).count(); \
      if (timeElapsed > nccl_nonblocking_timeout()) {                                                                 \
        std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" +                                         \
            std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +                                       \
            "\n" + getNcclErrorDetailStr(result, failureReason);                                                      \
        TORCH_CHECK_WITH(DistBackendError, false, err);                                                               \
      }                                                                                                               \
    }                                                                                                                 \
    ncclCommGetAsyncError(comm, &result);                                                                             \
  }                                                                                                                   \
  if (result != ncclSuccess) {                                                                                        \
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +                                               \
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(result) +                                           \
        "\n" + getNcclErrorDetailStr(result, failureReason);                                                          \
    TORCH_CHECK_WITH(DistBackendError, false, err);                                                                   \
  }

#define C10D_NCCL_CHECK_TIMEOUT_GROUPEND(cmd, comms_, failureReason)     \
  ncclResult_t state = cmd;                                                                                               \
  auto startTimepoint = std::chrono::steady_clock::now();                                                                 \
  if (state == ncclInProgress) {                                                                                          \
    for (const auto i : c10::irange(comms_.size())) {                                                                     \
      do {                                                                                                                \
        if (nccl_nonblocking_timeout() > 0) {                                                                             \
          auto currentTimepoint = std::chrono::steady_clock::now();                                                       \
          auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTimepoint - startTimepoint).count(); \
          if (timeElapsed > nccl_nonblocking_timeout()) {                                                                 \
            std::string err = "NCCL timeout in: " + std::string(__FILE__) + ":" +                                         \
                std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) +                                        \
                "\n" + getNcclErrorDetailStr(state, failureReason);                                                       \
            TORCH_CHECK_WITH(DistBackendError, false, err);                                                               \
          }                                                                                                               \
        }                                                                                                                 \
        ncclCommGetAsyncError(comms_[i]->getNcclComm(), &state);                                                          \
      } while (state == ncclInProgress);                                                                                  \
      if (state != ncclSuccess) {                                                                                         \
        break; /* fall through to failed case */                                                                          \
      }                                                                                                                   \
    }                                                                                                                     \
  }                                                                                                                       \
  if (state != ncclSuccess) {                                                                                             \
    std::string err = "NCCL error in: " + std::string(__FILE__) + ":" +                                                   \
        std::to_string(__LINE__) + ", " + ncclGetErrorWithVersion(state) +                                                \
        "\n" + getNcclErrorDetailStr(state, failureReason);                                                               \
    TORCH_CHECK_WITH(DistBackendError, false, err);                                                                       \
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

std::string getNcclVersion();
std::string ncclGetErrorWithVersion(ncclResult_t error);
bool nccl_use_nonblocking();
int nccl_nonblocking_timeout();

// Provides additional detail into NCCL error codes based on when these are
// thrown in the NCCL codebase.
std::string getNcclErrorDetailStr(
  ncclResult_t error,
  c10::optional<std::string> processGroupFailureReason = c10::nullopt);

// RAII wrapper for NCCL communicator
class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm),
        aborted_(false),
        ncclAsyncErr_(ncclSuccess),
        commFailureReason_(c10::nullopt) {}

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
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank), c10::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    return comm;
  }

#ifdef NCCL_HAS_COMM_NONBLOCKING
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      ncclConfig_t& config) {
    auto comm = std::make_shared<NCCLComm>();
    if (nccl_use_nonblocking()) {
      config.blocking = 0;
      C10D_NCCL_CHECK_TIMEOUT(
        ncclCommInitRankConfig(&(comm->ncclComm_), numRanks, commId, rank, &config), comm->ncclComm_, c10::nullopt);
    } else {
      C10D_NCCL_CHECK(
        ncclCommInitRankConfig(&(comm->ncclComm_), numRanks, commId, rank, &config), c10::nullopt);
    }
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    return comm;
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

    // Set true failure reason if provided by ProcessGroupNCCL (e.g. work
    // timeout)
    commFailureReason_ = commFailureReason;
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

  ncclResult_t checkForNcclError() {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (ncclAsyncErr_ != ncclSuccess) {
      return ncclAsyncErr_;
    }
    C10D_NCCL_CHECK(ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_), commFailureReason_);
    return ncclAsyncErr_;
#else
    // Always return success, if error checks are disabled.
    return ncclSuccess;
#endif
  }

 protected:
  ncclComm_t ncclComm_;
  // Unique nccl_id for this communicator.
  ncclUniqueId ncclId_;
  bool aborted_;
  ncclResult_t ncclAsyncErr_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupNCCL for
  // better error messaging.
  c10::optional<std::string> commFailureReason_;
};

// Helper that automatically cleans up premul sums.
struct ncclRedOpRAII {
  ncclRedOpRAII() = default;
  ncclRedOpRAII(ncclRedOp_t op) : op_(op) {}
  ncclRedOpRAII(ncclRedOp_t op, ncclComm_t comm) :
    op_(op), comm_(comm), premul_sum_(true) {}
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
  operator ncclRedOp_t() const { return op_; }
  ncclRedOp_t op_;
  ncclComm_t comm_;
  bool premul_sum_ = false;
};


} // namespace c10d

#endif // USE_C10D_NCCL
