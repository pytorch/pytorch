#pragma once

#include <nccl.h>
#include <memory>

// Error checking is enabled only for NCCL versions 2.4+ since ncclCommAbort()
// and ncclCommGetAsyncError() are not supported in earlier versions.
#if defined(NCCL_MAJOR) && (NCCL_MAJOR == 2) && defined(NCCL_MINOR) && \
    (NCCL_MINOR >= 4)
#define ENABLE_NCCL_ERROR_CHECKING
#elif defined(NCCL_MAJOR) && (NCCL_MAJOR >= 3)
#define ENABLE_NCCL_ERROR_CHECKING
#endif

#define C10D_NCCL_CHECK(cmd)                                              \
  do {                                                                    \
    ncclResult_t error = cmd;                                             \
    if (error != ncclSuccess) {                                           \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          std::string(ncclGetErrorString(error));                         \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

namespace c10d {

// RAII wrapper for NCCL communicator
class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm)
      : ncclComm_(ncclComm), aborted_(false), ncclAsyncErr_(ncclSuccess) {}

  NCCLComm() : NCCLComm(nullptr) {}

  ~NCCLComm() noexcept(false) {
    if (ncclComm_ && !aborted_) {
#ifdef ENABLE_NCCL_ERROR_CHECKING
      // Use ncclCommAbort instead of ncclCommDestroy here since
      // ncclCommDestroy could block forever waiting for work to complete on
      // the communicator.
      ncclCommAbort();
#else
      C10D_NCCL_CHECK(::ncclCommDestroy(ncclComm_));
#endif
    }
  }

  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();
    C10D_NCCL_CHECK(
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank));
    return comm;
  }

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Move constructable
  NCCLComm(NCCLComm&& other) {
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
  }

  // Move assignable
  NCCLComm& operator=(NCCLComm&& other) {
    std::swap(ncclComm_, other.ncclComm_);
    std::swap(aborted_, other.aborted_);
    std::swap(ncclAsyncErr_, other.ncclAsyncErr_);
    return *this;
  }

  ncclComm_t getNcclComm() {
    if (aborted_) {
      throw std::runtime_error("NCCL communicator was aborted.");
    }
    return ncclComm_;
  }

  void ncclCommAbort() {
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (aborted_) {
      // Should not abort twice.
      return;
    }

    C10D_NCCL_CHECK(::ncclCommAbort(ncclComm_));
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
    return aborted_;
  }

  ncclResult_t checkForNcclError() {
#ifdef ENABLE_NCCL_ERROR_CHECKING
    if (ncclAsyncErr_ != ncclSuccess) {
      return ncclAsyncErr_;
    }
    C10D_NCCL_CHECK(ncclCommGetAsyncError(ncclComm_, &ncclAsyncErr_));
    return ncclAsyncErr_;
#else
    // Always return success, if error checks are disabled.
    return ncclSuccess;
#endif
  }

 protected:
  ncclComm_t ncclComm_;
  bool aborted_;
  ncclResult_t ncclAsyncErr_;
};

} // namespace c10d
